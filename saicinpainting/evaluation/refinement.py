import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F

import numpy as np
import cv2

from kornia.geometry.transform import resize
from kornia.morphology import erosion, dilation
from saicinpainting.evaluation.data import pad_tensor_to_modulo
from saicinpainting.evaluation.utils import move_to_device

from tqdm import tqdm


def total_variation_masked(img: torch.Tensor, mask3: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    mask_h = mask3[:, :, 1:, :] * mask3[:, :, :-1, :]
    dw = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    mask_w = mask3[:, :, :, 1:] * mask3[:, :, :, :-1]

    num = (dh * mask_h).sum() + (dw * mask_w).sum()
    den = mask_h.sum() + mask_w.sum() + 1e-8
    return num / den


def boundary_ring_loss(
    img_ref: torch.Tensor,
    img_orig: torch.Tensor,
    mask1: torch.Tensor,
    kernel_size: int = 15
) -> torch.Tensor:
    """
    只在 mask 边界的一圈 ring 上，让 I_ref 靠近原图
    img_ref, img_orig: (B,3,H,W)
    mask1: (B,1,H,W)
    """
    device = img_ref.device
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    ekernel = torch.from_numpy(kernel.astype(bool)).float().to(device)

    # (B,1,H,W)
    dil = dilation(mask1, ekernel)
    ero = erosion(mask1, ekernel)
    ring = (dil - ero).clamp(0, 1.0)
    ring3 = ring.repeat(1, 3, 1, 1)

    diff = torch.abs(img_ref - img_orig) * ring3
    num = diff.sum()
    den = ring3.sum() + 1e-8
    return num / den


def refine_predict(
    batch: dict,
    inpainter: nn.Module,
    gpu_ids: str,
    modulo: int,
    n_iters: int,
    lr: float,
    min_side: int,
    max_scales: int,
    px_budget: int,
    lambda_data: float = 1.0,
    lambda_tv: float = 0.1,
):

    assert batch["image"].shape[0] == 1, "refiner only support batch_size=1"

    # 解析 device
    gpu_id_list = [gpuid for gpuid in gpu_ids.replace(" ", "").split(",") if gpuid.isdigit()]
    if len(gpu_id_list) > 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id_list[0]}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inpainter.to(device)
    inpainter.eval()


    h, w = batch["unpad_to_size"]
    h, w = h[0].item(), w[0].item()

    image = batch["image"][..., :h, :w]  # (1,3,H,W)
    mask1 = batch["mask"][..., :h, :w]   # (1,1,H,W)
    mask1 = (mask1 > 0).float()


    if h * w > px_budget:
        ratio = np.sqrt(px_budget / float(h * w))
        h_orig, w_orig = h, w
        h, w = int(h * ratio), int(w * ratio)
        print(f"Original image too large for refinement! Resizing {(h_orig, w_orig)} -> {(h, w)}")
        image = resize(image, (h, w), interpolation="bilinear", align_corners=False)
        mask1 = resize(mask1, (h, w), interpolation="nearest")
        mask1 = (mask1 > 0.5).float()

    image = pad_tensor_to_modulo(image, modulo)
    mask1 = pad_tensor_to_modulo(mask1, modulo)
    _, _, H_pad, W_pad = image.shape

    image = move_to_device(image, device)
    mask1 = move_to_device(mask1, device)

    mask3 = mask1.repeat(1, 3, 1, 1)           # (1,3,H,W)
    masked_image = image * (1.0 - mask3)
    gen_input = torch.cat([masked_image, mask1], dim=1)  # (1,4,H,W)  RGB+mask

    with torch.no_grad():
        pred = inpainter.generator(gen_input)  # (1,3,H,W)
    inpaint_init = mask3 * pred + (1.0 - mask3) * image  # (1,3,H,W)

    I_ref = inpaint_init.clone().detach().requires_grad_(True)
    optimizer = Adam([I_ref], lr=lr)

    pbar = tqdm(range(n_iters), leave=False, desc="Pixel refinement")
    for _ in pbar:
        optimizer.zero_grad()

        data_loss = torch.mean(torch.abs(
            I_ref[mask3 > 0.5] - inpaint_init[mask3 > 0.5]
        ))

        tv_loss = total_variation_masked(I_ref, mask3)

        loss = (
            lambda_data * data_loss +
            lambda_tv * tv_loss
        )

        pbar.set_description(f"Pixel refinement loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            I_ref.data[mask3 < 0.5] = image.data[mask3 < 0.5]

    I_ref = I_ref[:, :, :h, :w]


    I_ref = I_ref.detach().cpu()
    return I_ref
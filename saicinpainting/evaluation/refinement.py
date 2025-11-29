import torch
import torch.nn as nn
from torch.optim import Adam

from kornia.filters import gaussian_blur2d
from kornia.geometry.transform import resize
from kornia.morphology import erosion, dilation

from torch.nn import functional as F
import numpy as np
import cv2

from saicinpainting.evaluation.data import pad_tensor_to_modulo
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.modules.ffc import FFCResnetBlock
from saicinpainting.training.modules.pix2pixhd import ResnetBlock

from tqdm import tqdm

def _pyrdown(im: torch.Tensor, downsize: tuple = None):
    """Blur + downsample RGB image (B,3,H,W)."""
    if downsize is None:
        downsize = (im.shape[2] // 2, im.shape[3] // 2)
    assert im.shape[1] == 3
    im = gaussian_blur2d(im, kernel_size=(5, 5), sigma=(1.0, 1.0))
    im = F.interpolate(im, size=downsize, mode="bilinear", align_corners=False)
    return im


def _pyrdown_mask(
    mask: torch.Tensor,
    downsize: tuple = None,
    eps: float = 1e-8,
    blur_mask: bool = True,
    round_up: bool = True,
):
    """
    Downsample mask (B,1,H,W) with optional blur + binarize.
    """
    if downsize is None:
        downsize = (mask.shape[2] // 2, mask.shape[3] // 2)
    assert mask.shape[1] == 1

    if blur_mask:
        mask = gaussian_blur2d(mask, kernel_size=(5, 5), sigma=(1.0, 1.0))

    mask = F.interpolate(mask, size=downsize, mode="bilinear", align_corners=False)

    if round_up:
        mask[mask >= eps] = 1.0
        mask[mask < eps] = 0.0
    else:
        mask[mask >= 1.0 - eps] = 1.0
        mask[mask < 1.0 - eps] = 0.0

    return mask


def _erode_mask(mask: torch.Tensor, ekernel: torch.Tensor = None, eps: float = 1e-8):
    """Erode mask with kornia erosion, then hard-threshold."""
    if ekernel is not None:
        mask = erosion(mask, ekernel)
        mask[mask >= 1.0 - eps] = 1.0
        mask[mask < 1.0 - eps] = 0.0
    return mask


def _l1_loss(
    pred: torch.Tensor,
    pred_downscaled: torch.Tensor,
    ref: torch.Tensor,
    mask: torch.Tensor,
    mask_downscaled: torch.Tensor,
    image: torch.Tensor,
    on_pred: bool = True,
):
    """
    Original LaMa refinement L1:
    - outside mask: pred vs image
    - on downscaled mask: pred_downscaled vs ref_lower_res
    """
    loss = torch.mean(torch.abs(pred[mask < 1e-8] - image[mask < 1e-8]))
    if on_pred:
        loss = loss + torch.mean(
            torch.abs(pred_downscaled[mask_downscaled >= 1e-8] - ref[mask_downscaled >= 1e-8])
        )
    return loss


def _boundary_ring(mask_1: torch.Tensor, ekernel: torch.Tensor) -> torch.Tensor:
    """
    Compute a thin ring around mask boundary: (dilate - erode).
    mask_1: (B,1,H,W) 0/1
    """
    dil = dilation(mask_1, ekernel)
    ero = erosion(mask_1, ekernel)
    ring = (dil - ero).clamp(0.0, 1.0)
    return ring


def _edge_consistency_loss(
    pred: torch.Tensor,
    image: torch.Tensor,
    mask: torch.Tensor,
    ekernel: torch.Tensor,
) -> torch.Tensor:
    """
    Simple edge-aware loss: on boundary ring, pred should be close to image.
    """
    mask_1 = mask[:, :1]
    ring = _boundary_ring(mask_1, ekernel)  # (B,1,H,W)
    ring3 = ring.repeat(1, 3, 1, 1)
    diff = torch.abs(pred - image) * ring3
    denom = ring3.sum() + 1e-8
    return diff.sum() / denom


def _feat_mask_for_z(mask: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Downsample mask to feature map size (for masked latent updates).
    mask: (B,3,H,W) or (B,1,H,W)
    z:    (B,C,Hf,Wf)
    """
    if mask.shape[1] != 1:
        mask1 = mask[:, :1]
    else:
        mask1 = mask
    m = F.interpolate(mask1, size=z.shape[-2:], mode="nearest")
    return m


# ---------------------- core single-scale refinement ----------------------


def _infer(
    image: torch.Tensor,
    mask: torch.Tensor,
    forward_front: nn.Module,
    forward_rears: nn.Module,
    ref_lower_res: torch.Tensor,
    orig_shape: tuple,
    devices: list,
    scale_ind: int,
    n_iters: int = 15,
    lr: float = 0.002,
    # ablation & extra params
    ablation_mode: str = "R0",
    lambda_edge: float = 0.0,
    lambda_perc: float = 0.0,
    lambda_freq: float = 0.0,
    lr_local: float = 0.0,
    lr_global: float = 0.0,
    n_global_only: int = 0,
    lambda_struct: float = 0.0,
):
    """
    One-scale refinement. Different ablation modes (R0~R8) are controlled here.
    """

    # build masked input
    masked_image = image * (1 - mask)
    masked_image = torch.cat([masked_image, mask], dim=1)  # concat mask channel
    mask_rgb = mask.repeat(1, 3, 1, 1)                     # for blending + losses

    if ref_lower_res is not None:
        ref_lower_res = ref_lower_res.detach()

    # encode once
    with torch.no_grad():
        z1, z2 = forward_front(masked_image)

    # everything lives on first device initially
    mask_rgb = mask_rgb.to(devices[-1])
    image = image.to(devices[-1])

    z1 = z1.detach().to(devices[0])
    z2 = z2.detach().to(devices[0])
    z1.requires_grad_(True)
    z2.requires_grad_(True)

    # kernel for mask erosion / ring
    ekernel_np = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)).astype(bool)
    ekernel = torch.from_numpy(ekernel_np).float().to(devices[-1])

    # feature masks for masked-latent模式 (R3/R4/R5)
    feat_mask_z1 = _feat_mask_for_z(mask_rgb, z1).to(z1.device)
    feat_mask_z2 = _feat_mask_for_z(mask_rgb, z2).to(z2.device)

    if ablation_mode == "R7":
        # separate lrs for local/global
        if lr_local <= 0:
            lr_local = lr
        if lr_global <= 0:
            lr_global = lr
        optimizer = Adam(
            [
                {"params": [z1], "lr": lr_local},
                {"params": [z2], "lr": lr_global},
            ],
        )
    else:
        optimizer = Adam([z1, z2], lr=lr)

    pbar = tqdm(range(n_iters), leave=False)
    for idi in pbar:
        optimizer.zero_grad()

        # ---- forward through rear blocks (possibly across devices) ----
        input_feat = (z1, z2)
        for idd, forward_rear in enumerate(forward_rears):
            output_feat = forward_rear(input_feat)
            if idd < len(devices) - 1:
                midz1, midz2 = output_feat
                midz1 = midz1.to(devices[idd + 1])
                midz2 = midz2.to(devices[idd + 1])
                input_feat = (midz1, midz2)
            else:
                pred = output_feat  # (B,3,H,W) on last device

        if ref_lower_res is None:
            break

        losses = {}

        # ---------- R0: multi-scale L1 (always启用) ----------
        pred_cropped = pred[:, :, : orig_shape[0], : orig_shape[1]]
        pred_downscaled = _pyrdown(pred_cropped)
        mask_downscaled = _pyrdown_mask(
            mask[:, :1, : orig_shape[0], : orig_shape[1]],
            blur_mask=False,
            round_up=False,
        )
        mask_downscaled = _erode_mask(mask_downscaled, ekernel=ekernel)
        mask_downscaled = mask_downscaled.repeat(1, 3, 1, 1)

        losses["ms_l1"] = _l1_loss(
            pred,
            pred_downscaled,
            ref_lower_res,
            mask_rgb,
            mask_downscaled,
            image,
            on_pred=True,
        )

        if ablation_mode in ("R1", "R4", "R5"):
            loss_edge_img = _edge_consistency_loss(pred, image, mask, ekernel)
            losses["edge"] = lambda_edge * loss_edge_img

        if ablation_mode in ("R2", "R5") and lambda_perc > 0.0:
            ref_up = F.interpolate(
                ref_lower_res,
                size=orig_shape,
                mode="bilinear",
                align_corners=False,
            )

            m3 = mask_rgb[:, :, : orig_shape[0], : orig_shape[1]]
            diff_perc = torch.abs(pred_cropped - ref_up) * m3
            loss_perc = diff_perc.sum() / (m3.sum() + 1e-8)
            losses["perc"] = lambda_perc * loss_perc

        if ablation_mode in ("R6", "R8") and lambda_freq > 0.0:
            ref_up = F.interpolate(
                ref_lower_res,
                size=orig_shape,
                mode="bilinear",
                align_corners=False,
            )
            pred_fft = torch.fft.rfft2(pred_cropped, norm="ortho")
            ref_fft = torch.fft.rfft2(ref_up, norm="ortho")

            freq_mask = mask[:, :1, : orig_shape[0], : orig_shape[1]]
            freq_mask = F.interpolate(
                freq_mask,
                size=pred_fft.shape[-2:],
                mode="nearest",
            )
            freq_mask = (freq_mask >= 1e-8).float()
            freq_mask = freq_mask.expand_as(pred_fft)

            loss_freq = (torch.abs(pred_fft - ref_fft) * freq_mask).sum() / (
                freq_mask.sum() + 1e-8
            )
            losses["freq"] = lambda_freq * loss_freq

        if ablation_mode in ("R8",) and lambda_struct > 0.0:
            img_cropped = image[:, :, : orig_shape[0], : orig_shape[1]]
            img_down = _pyrdown(img_cropped)
            not_mask = 1.0 - mask_downscaled
            diff_struct = torch.abs(pred_downscaled - img_down) * not_mask
            loss_struct = diff_struct.sum() / (not_mask.sum() + 1e-8)
            losses["struct"] = lambda_struct * loss_struct


        loss = sum(losses.values())
        pbar.set_description(
            f"[{ablation_mode}] scale {scale_ind+1} loss: {loss.item():.4f}"
        )

        if idi < n_iters - 1:
            loss.backward()


            if ablation_mode == "R7" and idi < n_global_only:
                if z1.grad is not None:
                    z1.grad.zero_()

            if ablation_mode in ("R3", "R4", "R5"):
                if z1.grad is not None:
                    z1.grad = z1.grad * feat_mask_z1
                if z2.grad is not None:
                    z2.grad = z2.grad * feat_mask_z2

            optimizer.step()

            del pred_downscaled
            del loss
            del pred

    # blend with image
    inpainted = mask_rgb * pred + (1 - mask_rgb) * image
    inpainted = inpainted.detach().cpu()
    return inpainted


# ---------------------- pyramid construction ----------------------


def _get_image_mask_pyramid(
    batch: dict, min_side: int, max_scales: int, px_budget: int
):
    """
    Build image/mask pyramid (list of images & masks), smallest scale first.
    """

    assert batch["image"].shape[0] == 1, "refiner works on only batches of size 1!"

    h, w = batch["unpad_to_size"]
    h, w = h[0].item(), w[0].item()

    image = batch["image"][..., :h, :w]
    mask = batch["mask"][..., :h, :w]

    # limit resolution by pixel budget
    if h * w > px_budget:
        ratio = np.sqrt(px_budget / float(h * w))
        h_orig, w_orig = h, w
        h, w = int(h * ratio), int(w * ratio)
        print(
            f"Original image too large for refinement! Resizing {(h_orig, w_orig)} to {(h, w)}..."
        )
        image = resize(image, (h, w), interpolation="bilinear", align_corners=False)
        mask = resize(mask, (h, w), interpolation="bilinear", align_corners=False)
        mask[mask > 1e-8] = 1.0

    breadth = min(h, w)
    n_scales = min(1 + int(round(max(0, np.log2(breadth / min_side)))), max_scales)

    ls_images = [image]
    ls_masks = [mask]

    for _ in range(n_scales - 1):
        image_p = _pyrdown(ls_images[-1])
        mask_p = _pyrdown_mask(ls_masks[-1])
        ls_images.append(image_p)
        ls_masks.append(mask_p)

    # smallest resolution first
    return ls_images[::-1], ls_masks[::-1]


# ---------------------- public API ----------------------


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
    # new ablation config
    ablation_mode: str = "R0",
    lambda_edge: float = 0.0,
    lambda_perc: float = 0.0,
    lambda_freq: float = 0.0,
    lr_local: float = 0.0,
    lr_global: float = 0.0,
    n_global_only: int = 0,
    lambda_struct: float = 0.0,
):
    """
    External entry point called from predict.py
    """

    assert not inpainter.training
    assert not inpainter.add_noise_kwargs
    assert inpainter.concat_mask

    # parse gpu list
    gpu_ids_list = [
        f"cuda:{gpuid}"
        for gpuid in gpu_ids.replace(" ", "").split(",")
        if gpuid.isdigit()
    ]
    devices = [torch.device(gpu_id) for gpu_id in gpu_ids_list]

    # count resnet blocks and find first index
    n_resnet_blocks = 0
    first_resblock_ind = 0
    found_first_resblock = False
    for idl, module in enumerate(inpainter.generator.model):
        if isinstance(module, (FFCResnetBlock, ResnetBlock)):
            n_resnet_blocks += 1
            found_first_resblock = True
        elif not found_first_resblock:
            first_resblock_ind += 1

    resblocks_per_gpu = max(1, n_resnet_blocks // len(gpu_ids_list))

    # split generator into front + several rear segments
    forward_front = inpainter.generator.model[0:first_resblock_ind]
    forward_front.to(devices[0])

    forward_rears = []
    for idd in range(len(gpu_ids_list)):
        if idd < len(gpu_ids_list) - 1:
            rear = inpainter.generator.model[
                first_resblock_ind + resblocks_per_gpu * idd : first_resblock_ind
                + resblocks_per_gpu * (idd + 1)
            ]
        else:
            rear = inpainter.generator.model[first_resblock_ind + resblocks_per_gpu * idd :]
        rear.to(devices[idd])
        forward_rears.append(rear)

    # build pyramid
    ls_images, ls_masks = _get_image_mask_pyramid(
        batch, min_side=min_side, max_scales=max_scales, px_budget=px_budget
    )

    image_inpainted = None
    for ids, (image, mask) in enumerate(zip(ls_images, ls_masks)):
        orig_shape = image.shape[2:]

        image = pad_tensor_to_modulo(image, modulo)
        mask = pad_tensor_to_modulo(mask, modulo)
        mask[mask >= 1e-8] = 1.0
        mask[mask < 1e-8] = 0.0

        image = move_to_device(image, devices[0])
        mask = move_to_device(mask, devices[0])

        if image_inpainted is not None:
            image_inpainted = move_to_device(image_inpainted, devices[-1])

        image_inpainted = _infer(
            image,
            mask,
            forward_front,
            forward_rears,
            image_inpainted,
            orig_shape,
            devices,
            ids,
            n_iters=n_iters,
            lr=lr,
            ablation_mode=ablation_mode,
            lambda_edge=lambda_edge,
            lambda_perc=lambda_perc,
            lambda_freq=lambda_freq,
            lr_local=lr_local,
            lr_global=lr_global,
            n_global_only=n_global_only,
            lambda_struct=lambda_struct,
        )

        image_inpainted = image_inpainted[:, :, : orig_shape[0], : orig_shape[1]]

        # free
        image = image.detach().cpu()
        mask = mask.detach().cpu()

    return image_inpainted


import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from kornia.filters import gaussian_blur2d
from kornia.geometry.transform import resize
from kornia.morphology import erosion, dilation
from torch.nn import functional as F
import numpy as np
import cv2
from torchvision import models

from saicinpainting.evaluation.data import pad_tensor_to_modulo
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.modules.ffc import FFCResnetBlock
from saicinpainting.training.modules.pix2pixhd import ResnetBlock

from tqdm import tqdm


def _pyrdown(im : torch.Tensor, downsize : tuple=None):
    """downscale the image"""
    if downsize is None:
        downsize = (im.shape[2]//2, im.shape[3]//2)
    assert im.shape[1] == 3, "Expected shape for the input to be (n,3,height,width)"
    im = gaussian_blur2d(im, kernel_size=(5,5), sigma=(1.0,1.0))
    im = F.interpolate(im, size=downsize, mode='bilinear', align_corners=False)
    return im

def _pyrdown_mask(mask : torch.Tensor, downsize : tuple=None, eps : float=1e-8, blur_mask : bool=True, round_up : bool=True):
    """downscale the mask tensor

    Parameters
    ----------
    mask : torch.Tensor
        mask of size (B, 1, H, W)
    downsize : tuple, optional
        size to downscale to. If None, image is downscaled to half, by default None
    eps : float, optional
        threshold value for binarizing the mask, by default 1e-8
    blur_mask : bool, optional
        if True, apply gaussian filter before downscaling, by default True
    round_up : bool, optional
        if True, values above eps are marked 1, else, values below 1-eps are marked 0, by default True

    Returns
    -------
    torch.Tensor
        downscaled mask
    """

    if downsize is None:
        downsize = (mask.shape[2]//2, mask.shape[3]//2)
    assert mask.shape[1] == 1, "Expected shape for the input to be (n,1,height,width)"
    if blur_mask == True:
        mask = gaussian_blur2d(mask, kernel_size=(5,5), sigma=(1.0,1.0))
        mask = F.interpolate(mask, size=downsize,  mode='bilinear', align_corners=False)
    else:
        mask = F.interpolate(mask, size=downsize,  mode='bilinear', align_corners=False)
    if round_up:
        mask[mask>=eps] = 1
        mask[mask<eps] = 0
    else:
        mask[mask>=1.0-eps] = 1
        mask[mask<1.0-eps] = 0
    return mask

def _erode_mask(mask : torch.Tensor, ekernel : torch.Tensor=None, eps : float=1e-8):
    """erode the mask, and set gray pixels to 0"""
    if ekernel is not None:
        mask = erosion(mask, ekernel)
        mask[mask>=1.0-eps] = 1
        mask[mask<1.0-eps] = 0
    return mask


def _l1_loss(
    pred : torch.Tensor, pred_downscaled : torch.Tensor, ref : torch.Tensor,
    mask : torch.Tensor, mask_downscaled : torch.Tensor,
    image : torch.Tensor, on_pred : bool=True
    ):
    """l1 loss on src pixels, and downscaled predictions if on_pred=True"""
    loss = torch.mean(torch.abs(pred[mask<1e-8] - image[mask<1e-8]))
    if on_pred:
        loss += torch.mean(torch.abs(pred_downscaled[mask_downscaled>=1e-8] - ref[mask_downscaled>=1e-8]))
    return loss


_PERCEPTUAL_NET = None
def get_perceptual_net(device):
    global _PERCEPTUAL_NET
    if _PERCEPTUAL_NET is None:
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        except Exception:
            vgg = models.vgg16(pretrained=True)
        vgg_features = vgg.features[:16]
        vgg_features.eval()
        for p in vgg_features.parameters():
            p.requires_grad = False
        _PERCEPTUAL_NET = vgg_features.to(device)
    return _PERCEPTUAL_NET


def _infer(
    image : torch.Tensor, mask : torch.Tensor,
    forward_front : nn.Module, forward_rears : nn.Module,
    ref_lower_res : torch.Tensor, orig_shape : tuple, devices : list,
    scale_ind : int, n_iters : int=15, lr : float=0.002, ablation_mode="R0",lambda_edge=1.0,lambda_perc: float = 0.0):
    """Performs inference with refinement at a given scale.

    Parameters
    ----------
    image : torch.Tensor
        input image to be inpainted, of size (1,3,H,W)
    mask : torch.Tensor
        input inpainting mask, of size (1,1,H,W)
    forward_front : nn.Module
        the front part of the inpainting network
    forward_rears : nn.Module
        the rear part of the inpainting network
    ref_lower_res : torch.Tensor
        the inpainting at previous scale, used as reference image
    orig_shape : tuple
        shape of the original input image before padding
    devices : list
        list of available devices
    scale_ind : int
        the scale index
    n_iters : int, optional
        number of iterations of refinement, by default 15
    lr : float, optional
        learning rate, by default 0.002

    Returns
    -------
    torch.Tensor
        inpainted image
    """
    masked_image = image * (1 - mask)
    masked_image = torch.cat([masked_image, mask], dim=1)

    mask = mask.repeat(1,3,1,1)
    if ref_lower_res is not None:
        ref_lower_res = ref_lower_res.detach()
    with torch.no_grad():
        z1,z2 = forward_front(masked_image)
    # Inference
    mask = mask.to(devices[-1])
    ekernel = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)).astype(bool)).float()
    ekernel = ekernel.to(devices[-1])
    image = image.to(devices[-1])
    z1, z2 = z1.detach().to(devices[0]), z2.detach().to(devices[0])
    z1.requires_grad, z2.requires_grad = True, True

    optimizer = Adam([z1,z2], lr=lr)

    pbar = tqdm(range(n_iters), leave=False)
    for idi in pbar:
        optimizer.zero_grad()
        input_feat = (z1,z2)
        for idd, forward_rear in enumerate(forward_rears):
            output_feat = forward_rear(input_feat)
            if idd < len(devices) - 1:
                midz1, midz2 = output_feat
                midz1, midz2 = midz1.to(devices[idd+1]), midz2.to(devices[idd+1])
                input_feat = (midz1, midz2)
            else:
                pred = output_feat

        if ref_lower_res is None:
            break
        losses = {}
        ######################### multi-scale #############################
        # scaled loss with downsampler
        pred_downscaled = _pyrdown(pred[:,:,:orig_shape[0],:orig_shape[1]])
        mask_downscaled = _pyrdown_mask(mask[:,:1,:orig_shape[0],:orig_shape[1]], blur_mask=False, round_up=False)
        mask_downscaled = _erode_mask(mask_downscaled, ekernel=ekernel)
        mask_downscaled = mask_downscaled.repeat(1,3,1,1)
        losses["ms_l1"] = _l1_loss(pred, pred_downscaled, ref_lower_res, mask, mask_downscaled, image, on_pred=True)



        # ---------- R1: Boundary-aware L1 Loss ----------
        if ablation_mode in ("R1", "R5"):
            print("R1 running")
            # mask_full = full resolution mask (1 channel)
            mask_full = mask[:, :1, :orig_shape[0], :orig_shape[1]]  # shape (B,1,H,W)
            ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            ring_kernel = torch.from_numpy(ring_kernel).float().to(mask_full.device)
            # dilation / erosion
            mask_dilate = dilation(mask_full, ring_kernel)
            mask_erode  = erosion(mask_full,  ring_kernel)
            ring = (mask_dilate - mask_erode).clamp(0, 1)
            ring = ring.repeat(1,3,1,1)
            # L1(pred, image) but only on ring region
            pred_cropped  = pred[:, :, :orig_shape[0], :orig_shape[1]]
            img_cropped   = image[:, :, :orig_shape[0], :orig_shape[1]]
            edge_loss = torch.mean(torch.abs(
                pred_cropped[ring > 0.5] - img_cropped[ring > 0.5]
            ))
            losses["edge_l1"] = lambda_edge * edge_loss




        # ---------- R2: Perceptual loss ----------
        if ablation_mode in ("R2", "R4") and lambda_perc > 0.0:
            print("R3 running")
            ref_up = F.interpolate(
                ref_lower_res, size=orig_shape,
                mode='bilinear', align_corners=False
            )  # (B,3,H,W)
            pred_cropped = pred[:, :, :orig_shape[0], :orig_shape[1]]   # (B,3,H,W)
            mask_full    = mask[:, :1, :orig_shape[0], :orig_shape[1]]  # (B,1,H,W)

            perc_net = get_perceptual_net(pred_cropped.device)
            feat_pred = perc_net(pred_cropped)   # (B,Cf,Hf,Wf)
            with torch.no_grad():
                feat_ref  = perc_net(ref_up)     # (B,Cf,Hf,Wf)

            feat_mask = F.interpolate(
                mask_full, size=feat_pred.shape[-2:], mode='nearest'
            )                                # (B,1,Hf,Wf)
            feat_mask = (feat_mask >= 1e-8).float()
            feat_mask = feat_mask.expand_as(feat_pred)   # (B,Cf,Hf,Wf)

            diff = torch.abs(feat_pred - feat_ref) * feat_mask
            perc_loss = diff.sum() / (feat_mask.sum() + 1e-8)
            losses["perceptual"] = lambda_perc * perc_loss




        loss = sum(losses.values())
        pbar.set_description("Refining scale {} using scale {} ...current loss: {:.4f}".format(scale_ind+1, scale_ind, loss.item()))
        if idi < n_iters - 1:
            loss.backward()

            # ---------- R3: Masked latent refinement ----------
            if ablation_mode in ("R3", "R4","R5"):
                print("R3 running")
                mask_full = mask[:, :1, :orig_shape[0], :orig_shape[1]].to(z1.device)
                feat_mask = F.interpolate(
                    mask_full, size=z1.shape[-2:], mode='nearest'
                )
                feat_mask = (feat_mask >= 1e-8).float()
                if z1.grad is not None:
                    z1.grad *= feat_mask
                if z2.grad is not None:
                    z2.grad *= feat_mask


            optimizer.step()
            del pred_downscaled
            del loss
            del pred
    # "pred" is the prediction after Plug-n-Play module
    inpainted = mask * pred + (1 - mask) * image
    inpainted = inpainted.detach().cpu()
    return inpainted

def _get_image_mask_pyramid(batch : dict, min_side : int, max_scales : int, px_budget : int):
    """Build the image mask pyramid

    Parameters
    ----------
    batch : dict
        batch containing image, mask, etc
    min_side : int
        minimum side length to limit the number of scales of the pyramid
    max_scales : int
        maximum number of scales allowed
    px_budget : int
        the product H*W cannot exceed this budget, because of resource constraints

    Returns
    -------
    tuple
        image-mask pyramid in the form of list of images and list of masks
    """

    assert batch['image'].shape[0] == 1, "refiner works on only batches of size 1!"

    h, w = batch['unpad_to_size']
    h, w = h[0].item(), w[0].item()

    image = batch['image'][...,:h,:w]
    mask = batch['mask'][...,:h,:w]
    if h*w > px_budget:
        #resize
        ratio = np.sqrt(px_budget / float(h*w))
        h_orig, w_orig = h, w
        h,w = int(h*ratio), int(w*ratio)
        print(f"Original image too large for refinement! Resizing {(h_orig,w_orig)} to {(h,w)}...")
        image = resize(image, (h,w),interpolation='bilinear', align_corners=False)
        mask = resize(mask, (h,w),interpolation='bilinear', align_corners=False)
        mask[mask>1e-8] = 1
    breadth = min(h,w)
    n_scales = min(1 + int(round(max(0,np.log2(breadth / min_side)))), max_scales)
    ls_images = []
    ls_masks = []

    ls_images.append(image)
    ls_masks.append(mask)

    for _ in range(n_scales - 1):
        image_p = _pyrdown(ls_images[-1])
        mask_p = _pyrdown_mask(ls_masks[-1])
        ls_images.append(image_p)
        ls_masks.append(mask_p)
    # reverse the lists because we want the lowest resolution image as index 0
    return ls_images[::-1], ls_masks[::-1]

def refine_predict(
    batch : dict, inpainter : nn.Module, gpu_ids : str,
    modulo : int, n_iters : int, lr : float, min_side : int,
    max_scales : int, px_budget : int
    , ablation_mode: str = "R0", lambda_edge = None,lambda_perc: float = 0.0):
    """Refines the inpainting of the network

    Parameters
    ----------
    batch : dict
        image-mask batch, currently we assume the batchsize to be 1
    inpainter : nn.Module
        the inpainting neural network
    gpu_ids : str
        the GPU ids of the machine to use. If only single GPU, use: "0,"
    modulo : int
        pad the image to ensure dimension % modulo == 0
    n_iters : int
        number of iterations of refinement for each scale
    lr : float
        learning rate
    min_side : int
        all sides of image on all scales should be >= min_side / sqrt(2)
    max_scales : int
        max number of downscaling scales for the image-mask pyramid
    px_budget : int
        pixels budget. Any image will be resized to satisfy height*width <= px_budget

    Returns
    -------
    torch.Tensor
        inpainted image of size (1,3,H,W)
    """

    assert not inpainter.training
    assert not inpainter.add_noise_kwargs
    assert inpainter.concat_mask

    gpu_ids = [f'cuda:{gpuid}' for gpuid in gpu_ids.replace(" ","").split(",") if gpuid.isdigit()]
    n_resnet_blocks = 0
    first_resblock_ind = 0
    found_first_resblock = False
    for idl in range(len(inpainter.generator.model)):
        if isinstance(inpainter.generator.model[idl], FFCResnetBlock) or isinstance(inpainter.generator.model[idl], ResnetBlock):
            n_resnet_blocks += 1
            found_first_resblock = True
        elif not found_first_resblock:
            first_resblock_ind += 1
    resblocks_per_gpu = n_resnet_blocks // len(gpu_ids)

    devices = [torch.device(gpu_id) for gpu_id in gpu_ids]

    # split the model into front, and rear parts
    forward_front = inpainter.generator.model[0:first_resblock_ind]
    forward_front.to(devices[0])
    forward_rears = []
    for idd in range(len(gpu_ids)):
        if idd < len(gpu_ids) - 1:
            forward_rears.append(inpainter.generator.model[first_resblock_ind + resblocks_per_gpu*(idd):first_resblock_ind+resblocks_per_gpu*(idd+1)])
        else:
            forward_rears.append(inpainter.generator.model[first_resblock_ind + resblocks_per_gpu*(idd):])
        forward_rears[idd].to(devices[idd])

    ls_images, ls_masks = _get_image_mask_pyramid(
        batch,
        min_side,
        max_scales,
        px_budget
        )
    image_inpainted = None

    for ids, (image, mask) in enumerate(zip(ls_images, ls_masks)):
        orig_shape = image.shape[2:]
        image = pad_tensor_to_modulo(image, modulo)
        mask = pad_tensor_to_modulo(mask, modulo)
        mask[mask >= 1e-8] = 1.0
        mask[mask < 1e-8] = 0.0
        image, mask = move_to_device(image, devices[0]), move_to_device(mask, devices[0])
        if image_inpainted is not None:
            image_inpainted = move_to_device(image_inpainted, devices[-1])


        # image_inpainted = _infer(image, mask, forward_front, forward_rears, image_inpainted, orig_shape, devices, ids, n_iters, lr)


        # ---------- R1: Boundary-aware L1 Loss ----------
        image_inpainted = _infer(
            image, mask, forward_front, forward_rears,
            image_inpainted, orig_shape, devices, ids,
            n_iters, lr,
            ablation_mode=ablation_mode,
            # lambda_edge=lambda_edge,
            lambda_perc=lambda_perc,
        )


        image_inpainted = image_inpainted[:,:,:orig_shape[0], :orig_shape[1]]
        # detach everything to save resources
        image = image.detach().cpu()
        mask = mask.detach().cpu()

    return image_inpainted

#
# import torch
# import torch.nn as nn
# from torch.optim import Adam
# from torch.nn import functional as F
#
# import numpy as np
# import cv2
#
# from kornia.geometry.transform import resize
# from kornia.morphology import erosion, dilation
# from saicinpainting.evaluation.data import pad_tensor_to_modulo
# from saicinpainting.evaluation.utils import move_to_device
#
# from tqdm import tqdm
#
#
# # ------------------------------
# # 小工具：TV loss（只在 mask 区）
# # ------------------------------
# def total_variation_masked(img: torch.Tensor, mask3: torch.Tensor) -> torch.Tensor:
#     """
#     img:  (B,3,H,W)
#     mask3: (B,3,H,W) 取值 0/1，只在 1 的区域计算 TV
#     """
#     # H 方向梯度
#     dh = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
#     mask_h = mask3[:, :, 1:, :] * mask3[:, :, :-1, :]
#     # W 方向梯度
#     dw = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
#     mask_w = mask3[:, :, :, 1:] * mask3[:, :, :, :-1]
#
#     num = (dh * mask_h).sum() + (dw * mask_w).sum()
#     den = mask_h.sum() + mask_w.sum() + 1e-8
#     return num / den
#
#
# # ------------------------------
# # 小工具：边界带 loss
# # ------------------------------
# def boundary_ring_loss(
#     img_ref: torch.Tensor,
#     img_orig: torch.Tensor,
#     mask1: torch.Tensor,
#     kernel_size: int = 15
# ) -> torch.Tensor:
#     """
#     只在 mask 边界的一圈 ring 上，让 I_ref 靠近原图
#     img_ref, img_orig: (B,3,H,W)
#     mask1: (B,1,H,W)
#     """
#     device = img_ref.device
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
#     ekernel = torch.from_numpy(kernel.astype(bool)).float().to(device)
#
#     # (B,1,H,W)
#     dil = dilation(mask1, ekernel)
#     ero = erosion(mask1, ekernel)
#     ring = (dil - ero).clamp(0, 1.0)
#     ring3 = ring.repeat(1, 3, 1, 1)
#
#     diff = torch.abs(img_ref - img_orig) * ring3
#     num = diff.sum()
#     den = ring3.sum() + 1e-8
#     return num / den
#
#
# # ------------------------------
# # 主函数：像素空间 refinement
# # ------------------------------
# def refine_predict(
#     batch: dict,
#     inpainter: nn.Module,
#     gpu_ids: str,
#     modulo: int,
#     n_iters: int,
#     lr: float,
#     min_side: int,
#     max_scales: int,
#     px_budget: int,
#     # 以下是像素 refinement 的三个权重，可在 config 里调
#     lambda_data: float = 1.0,
#     lambda_tv: float = 0.1,
#     lambda_edge: float = 1.0,
# ):
#     """
#     Pixel-space refinement（方法 1）
#
#     - 先用 LaMa generator 得到初始 inpaint 结果
#     - 然后只对 mask 区域像素做梯度下降，优化:
#         lambda_data * |I_ref - I_init|  (在 mask 区)
#       + lambda_tv   * TV(I_ref)        (在 mask 区)
#       + lambda_edge * |I_ref - image|  (在边界 ring 区)
#     - 非 mask 区域始终等于原图
#
#     参数说明大致沿用原版 refiner 的接口，便于和 predict.py / config 对接。
#     """
#
#     assert batch["image"].shape[0] == 1, "refiner 目前只支持 batch_size=1"
#
#     # 解析 device
#     gpu_id_list = [gpuid for gpuid in gpu_ids.replace(" ", "").split(",") if gpuid.isdigit()]
#     if len(gpu_id_list) > 0 and torch.cuda.is_available():
#         device = torch.device(f"cuda:{gpu_id_list[0]}")
#     else:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     inpainter.to(device)
#     inpainter.eval()
#
#     # ------------------------------
#     # 1) 取出原图 / mask，裁掉 pad 区域
#     # ------------------------------
#     h, w = batch["unpad_to_size"]
#     h, w = h[0].item(), w[0].item()
#
#     image = batch["image"][..., :h, :w]  # (1,3,H,W)
#     mask1 = batch["mask"][..., :h, :w]   # (1,1,H,W)
#     mask1 = (mask1 > 0).float()
#
#     # 可选：太大就按 px_budget 等比缩小（和原版逻辑类似）
#     if h * w > px_budget:
#         ratio = np.sqrt(px_budget / float(h * w))
#         h_orig, w_orig = h, w
#         h, w = int(h * ratio), int(w * ratio)
#         print(f"Original image too large for refinement! Resizing {(h_orig, w_orig)} -> {(h, w)}")
#         image = resize(image, (h, w), interpolation="bilinear", align_corners=False)
#         mask1 = resize(mask1, (h, w), interpolation="nearest")
#         mask1 = (mask1 > 0.5).float()
#
#     # pad 到 modulo
#     image = pad_tensor_to_modulo(image, modulo)
#     mask1 = pad_tensor_to_modulo(mask1, modulo)
#     _, _, H_pad, W_pad = image.shape
#
#     # 移到 device
#     image = move_to_device(image, device)
#     mask1 = move_to_device(mask1, device)
#
#     # ------------------------------
#     # 2) 用 LaMa generator 跑一次普通 inpaint 得到 I_init
#     # ------------------------------
#     mask3 = mask1.repeat(1, 3, 1, 1)           # (1,3,H,W)
#     masked_image = image * (1.0 - mask3)       # 把 mask 区盖掉
#     gen_input = torch.cat([masked_image, mask1], dim=1)  # (1,4,H,W)  RGB+mask
#
#     with torch.no_grad():
#         pred = inpainter.generator(gen_input)  # (1,3,H,W)
#     inpaint_init = mask3 * pred + (1.0 - mask3) * image  # (1,3,H,W)
#
#     # ------------------------------
#     # 3) 像素空间优化：只让 mask 区的像素可学习
#     # ------------------------------
#     I_ref = inpaint_init.clone().detach().requires_grad_(True)
#     optimizer = Adam([I_ref], lr=lr)
#
#     pbar = tqdm(range(n_iters), leave=False, desc="Pixel refinement")
#     for _ in pbar:
#         optimizer.zero_grad()
#
#         # 在 mask 区域与初始 inpaint 保持接近
#         data_loss = torch.mean(torch.abs(
#             I_ref[mask3 > 0.5] - inpaint_init[mask3 > 0.5]
#         ))
#
#         # mask 区域的 TV 平滑
#         tv_loss = total_variation_masked(I_ref, mask3)
#
#         # 边界 ring 区域与原图一致
#         edge_loss = boundary_ring_loss(I_ref, image, mask1)
#
#         loss = (
#             lambda_data * data_loss +
#             lambda_tv * tv_loss +
#             lambda_edge * edge_loss
#         )
#
#         pbar.set_description(f"Pixel refinement loss: {loss.item():.4f}")
#         loss.backward()
#         optimizer.step()
#
#         # 非 mask 区域始终强制等于原图
#         with torch.no_grad():
#             I_ref.data[mask3 < 0.5] = image.data[mask3 < 0.5]
#
#     # ------------------------------
#     # 4) 去掉 padding / resize 回原尺度
#     # ------------------------------
#     I_ref = I_ref[:, :, :h, :w]   # 去 padding
#
#     # 如果前面有 resize 过，这里可以按需要再放大回去
#     # 目前直接返回 resize 后的结果（和原 refiner 行为一致）
#
#     I_ref = I_ref.detach().cpu()
#     return I_ref

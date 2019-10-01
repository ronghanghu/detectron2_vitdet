import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

__all__ = ["paste_masks_in_image"]


BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit


def paste_masks_in_image(masks, boxes, image_shape, threshold=0.5):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes): A Boxes of length Bimg. boxes.tensor[i] and masks[i] correspond
            to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.

    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
            number of detected object instances and Himage, Wimage are the image width
            and height. img_masks[i] is a binary mask for object instance i.
    """
    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    if len(masks) == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)

    boxes = boxes.tensor

    device = boxes.device
    N, mask_h, mask_w = masks.shape
    assert len(boxes) == N, boxes.shape
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    img_h, img_w = image_shape

    if device.type == "cpu":
        # on CPU, paste the masks one by one, by sampling coordinates in the box
        x0, y0 = x0[:, 0], y0[:, 0]
        x1, y1 = x1[:, 0], y1[:, 0]
        zero = torch.Tensor([0]).to(device=x0.device, dtype=torch.int32)
        x0_int = torch.max(zero, x0.floor().to(dtype=torch.int32) - 1)
        y0_int = torch.max(zero, y0.floor().to(dtype=torch.int32) - 1)
        x1_int = torch.min(zero + img_w, x1.ceil().to(dtype=torch.int32) + 1)
        y1_int = torch.min(zero + img_h, y1.ceil().to(dtype=torch.int32) + 1)

        img_masks = torch.zeros(N, img_h, img_w, device=masks.device, dtype=torch.uint8)
        for idx, mask in enumerate(masks):
            img_y = torch.arange(y0_int[idx], y1_int[idx], dtype=torch.float32, device=device) + 0.5
            img_x = torch.arange(x0_int[idx], x1_int[idx], dtype=torch.float32, device=device) + 0.5
            img_y = (img_y - y0[idx]) / (y1[idx] - y0[idx]) * 2 - 1
            img_x = (img_x - x0[idx]) / (x1[idx] - x0[idx]) * 2 - 1

            gy, gx = torch.meshgrid(img_y, img_x)
            ind = torch.stack([gx, gy], dim=-1).to(dtype=torch.float32, device=masks.device)
            # Use align_corners=False. See https://github.com/pytorch/pytorch/issues/20785
            res = F.grid_sample(
                mask[None, None, :, :].to(dtype=torch.float32),
                ind[None, :, :, :],
                align_corners=False,
            )
            if threshold >= 0:
                res = (res >= threshold).to(dtype=torch.uint8)
            else:
                res = (res * 255).to(dtype=torch.uint8)
            img_masks[idx, y0_int[idx] : y1_int[idx], x0_int[idx] : x1_int[idx]] = res
        return img_masks
    else:

        def process_chunk(masks_chunk, y0_chunk, y1_chunk, x0_chunk, x1_chunk):
            # On GPU, paste all masks together (up to chunk size)
            # by using the entire image to sample the masks
            # Compared to pasting them one by one,
            # this has more operations but is faster on COCO-scale dataset.
            N_chunk = masks_chunk.shape[0]
            img_y = torch.arange(0.0, img_h).to(device=device) + 0.5
            img_x = torch.arange(0.0, img_w).to(device=device) + 0.5
            img_y = (img_y - y0_chunk) / (y1_chunk - y0_chunk) * 2 - 1
            img_x = (img_x - x0_chunk) / (x1_chunk - x0_chunk) * 2 - 1
            # img_x, img_y have shapes (N_chunk, img_w), (N_chunk, img_h)

            gx = img_x[:, None, :].expand(N_chunk, img_h, img_w)
            gy = img_y[:, :, None].expand(N_chunk, img_h, img_w)
            grid = torch.stack([gx, gy], dim=3)

            img_masks = F.grid_sample(
                masks_chunk.to(dtype=torch.float32), grid, align_corners=False
            )
            if threshold >= 0:
                img_masks = (img_masks >= threshold).to(dtype=torch.uint8)
            else:
                # for visualization and debugging
                img_masks = (img_masks * 255).to(dtype=torch.uint8)
            return img_masks

        num_chunks = int(np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert num_chunks <= N, "Insufficient GPU memory; try increasing GPU_MEM_LIMIT"
        chunks = torch.chunk(torch.arange(N), num_chunks)
        img_masks = []
        for inds in chunks:
            img_masks.append(
                process_chunk(masks[inds, None, :, :], y0[inds], y1[inds], x0[inds], x1[inds])
            )
        img_masks = torch.cat(img_masks)[:, 0, :, :]
    return img_masks


# The below are the original paste function (from Detectron1) which has
# larger quantization error.
# It is faster on CPU, while the aligned one is faster on GPU thanks to grid_sample.


def paste_mask_in_image_old(mask, box, img_h, img_w, threshold):
    """
    Paste a single mask in an image.
    This is a per-box implementation of :func:`paste_masks_in_image`.
    This function has larger quantization error due to incorrect pixel
    modeling and is not used any more.

    Args:
        mask (Tensor): A tensor of shape (Hmask, Wmask) storing the mask of a single
            object instance. Values are in [0, 1].
        box (Tensor): A tensor of shape (4, ) storing the x0, y0, x1, y1 box corners
            of the object instance.
        img_h, img_w (int): Image height and width.
        threshold (float): Mask binarization threshold in [0, 1].

    Returns:
        im_mask (Tensor): The resized and binarized object mask pasted into the original
            image plane (a tensor of shape (img_h, img_w)).
    """
    # Conversion from continuous box coordinates to discrete pixel coordinates
    # via truncation (cast to int32). This determines which pixels to paste the
    # mask onto.
    box = box.to(dtype=torch.int32)  # Continuous to discrete coordinate conversion
    # An example (1D) box with continuous coordinates (x0=0.7, x1=4.3) will map to
    # a discrete coordinates (x0=0, x1=4). Note that box is mapped to 5 = x1 - x0 + 1
    # pixels (not x1 - x0 pixels).
    samples_w = box[2] - box[0] + 1  # Number of pixel samples, *not* geometric width
    samples_h = box[3] - box[1] + 1  # Number of pixel samples, *not* geometric height

    # Resample the mask from it's original grid to the new samples_w x samples_h grid
    mask = Image.fromarray(mask.cpu().numpy())
    mask = mask.resize((samples_w, samples_h), resample=Image.BILINEAR)
    mask = np.array(mask, copy=False)

    if threshold >= 0:
        mask = np.array(mask > threshold, dtype=np.uint8)
        mask = torch.from_numpy(mask)
    else:
        # for visualization and debugging, we also
        # allow it to return an unmodified mask
        mask = torch.from_numpy(mask * 255).to(torch.uint8)

    im_mask = torch.zeros((img_h, img_w), dtype=torch.uint8)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, img_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, img_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    return im_mask


# Our pixel modeling requires extrapolation for any continuous
# coordinate < 0.5 or > length - 0.5. When sampling pixels on the masks,
# we would like this extrapolation to be an interpolation between boundary values and zero,
# instead of using absolute zero or boundary values.
# Therefore `paste_mask_in_image_old` is often used with zero padding around the masks like this:
# masks, scale = pad_masks(masks[:, 0, :, :], 1)
# boxes = scale_boxes(boxes.tensor, scale)


def pad_masks(masks, padding):
    """
    Args:
        masks (tensor): A tensor of shape (B, M, M) representing B masks.
        padding (int): Number of cells to pad on all sides.

    Returns:
        The padded masks and the scale factor of the padding size / original size.
    """
    B = masks.shape[0]
    M = masks.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_masks = masks.new_zeros((B, M + pad2, M + pad2))
    padded_masks[:, padding:-padding, padding:-padding] = masks
    return padded_masks, scale


def scale_boxes(boxes, scale):
    """
    Args:
        boxes (tensor): A tensor of shape (B, 4) representing B boxes with 4
            coords representing the corners x0, y0, x1, y1,
        scale (float): The box scaling factor.

    Returns:
        Scaled boxes.
    """
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

    w_half *= scale
    h_half *= scale

    scaled_boxes = torch.zeros_like(boxes)
    scaled_boxes[:, 0] = x_c - w_half
    scaled_boxes[:, 2] = x_c + w_half
    scaled_boxes[:, 1] = y_c - h_half
    scaled_boxes[:, 3] = y_c + h_half
    return scaled_boxes

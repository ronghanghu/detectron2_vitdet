import numpy as np
import torch
from PIL import Image


# The infamous "+ 1" for box width and height dating back to the DPM days:
# https://github.com/rbgirshick/voc-dpm/blob/master/data/pascal_data.m#L72
_TO_REMOVE = 1  # See https://github.com/fairinternal/detectron2/issues/49


def paste_masks_in_image(masks, boxes, image_shape, threshold=0.5, padding=1):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Args:
        masks (tensor): Tensor of shape (Bimg, 1, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes): A Boxes of length Bimg. boxes.tensor[i] and masks[i] correspond
            to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.
        padding (int): Amount of padding to apply to the masks before pasting them. Padding
            is used to avoid "top hat" artifacts that can be caused by default non-zero padding
            that is implemented in some low-level image resizing implementations (such as opencv's
            cv2.resize function).

    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, 1, Himage, Wimage), where Bimg is the
            number of detected object instances and Himage, Wimage are the image width
            and height. img_masks[i] is a binary mask for object instance i.
    """
    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"

    assert boxes.mode == "xyxy"
    img_h, img_w = image_shape

    masks, scale = pad_masks(masks, padding)
    scaled_boxes = scale_boxes(boxes.tensor, scale)

    img_masks = [
        paste_mask_in_image(mask[0], box, img_h, img_w, threshold)
        for mask, box in zip(masks, scaled_boxes)
    ]
    if len(img_masks) > 0:
        img_masks = torch.stack(img_masks, dim=0)[:, None]
    else:
        img_masks = masks.new_empty((0, 1, img_h, img_w))
    return img_masks


def paste_mask_in_image(mask, box, img_h, img_w, threshold):
    """
    Paste a single mask in an image.

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
    # Quantize box to determine which pixels indices the mask will be pasted into.
    box = box.to(dtype=torch.int32)
    w = box[2] - box[0] + _TO_REMOVE
    h = box[3] - box[1] + _TO_REMOVE
    w = max(w, 1)
    h = max(h, 1)

    mask = Image.fromarray(mask.cpu().numpy())
    mask = mask.resize((w, h), resample=Image.BILINEAR)
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
    padded_masks = masks.new_zeros((B, 1, M + pad2, M + pad2))
    padded_masks[:, :, padding:-padding, padding:-padding] = masks
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

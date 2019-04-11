import torch
from torch.nn import functional as F

from detectron2.structures import Instances

from .roi_heads.paste_mask import paste_masks_in_image


def detector_postprocess(results, output_height, output_width):
    """
    Postprocess the output boxes.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will postprocess the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object will be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the postprocessed output from the model, based on the output resolution
    """
    # Faster on CPU, probably because paste_masks contains many cpu operations
    results = results.to(torch.device("cpu"))
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.tensor[:, 0::2] *= scale_x
    output_boxes.tensor[:, 1::2] *= scale_y
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        MASK_THRESHOLD = 0.5
        results.pred_masks = paste_masks_in_image(
            results.pred_masks,  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=MASK_THRESHOLD,
            padding=1,
        ).squeeze(1)

    if results.has("pred_keypoints"):
        results.pred_keypoints.tensor[:, :, 0] *= scale_x
        results.pred_keypoints.tensor[:, :, 1] *= scale_y

    return results


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Postprocess the output semantic segmentation logit predictions. Return semantic segmentation
    labels predicted for each pixel in the original resolution.

    The input images are often resized when entering semantic segmentator. Moreover, in same
    cases, they also padded inside segmentator to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentator in a different
    resolution from its inputs.

    After resizing the logits to the desired resolutions, argmax is applied to return semnantic
    segmentation classes predicted for each pixel.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentator is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmenation prediction (Tensor): A tensor of the shape
            (output_height, output_width) that contains per-pixel semantic segementation prediction.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(result, size=(output_height, output_width), mode="bilinear")[0]
    return result.argmax(dim=0)

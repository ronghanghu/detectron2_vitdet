import torch
from torch.nn import functional as F

from detectron2.layers import paste_masks_in_image
from detectron2.structures import Instances


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
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
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the postprocessed output from the model, based on the output resolution
    """
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
        results.pred_masks = paste_masks_in_image(
            results.pred_masks,  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        ).squeeze(1)

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    if results.has("densepose"):
        densepose_output = results.densepose
        output_boxes_xywh = output_boxes.tensor.clone()
        output_boxes_xywh[:, 2] -= output_boxes_xywh[:, 0]
        output_boxes_xywh[:, 3] -= output_boxes_xywh[:, 1]
        densepose_result = densepose_output.to_result(output_boxes_xywh)
        results.densepose = densepose_result

    return results


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Postprocess the output semantic segmentation logit predictions. Return semantic segmentation
    labels predicted for each pixel in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    After resizing the logits to the desired resolutions, argmax is applied to return semnantic
    segmentation classes predicted for each pixel.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmenation prediction (Tensor): A tensor of the shape
            (output_height, output_width) that contains per-pixel semantic segementation prediction.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result.argmax(dim=0)


def combine_semantic_and_instance_outputs(
    instance_results,
    semantic_results,
    overlap_threshold,
    stuff_area_limit,
    instances_confidence_threshold,
):
    """
    Implement a simple combining logic following
    https://github.com/cocodataset/panopticapi/blob/master/combine_semantic_and_instance_predictions.py
    to produce panoptic segmentation outputs.

    Args:
        instance_results: output of :func:`detector_postprocess`.
        semantic_results: output of :func:`sem_seg_postprocess`.

    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
    """
    panoptic_seg = torch.zeros_like(semantic_results)

    # sort instance outputs by scores
    sorted_inds = torch.argsort(-instance_results.scores)

    current_segment_id = 0
    segments_info = []

    instance_masks = instance_results.pred_masks.to(device=panoptic_seg.device, dtype=torch.bool)

    # Add instances one-by-one, check for overlaps with existing ones
    for inst_id in sorted_inds:
        if instance_results.scores[inst_id].item() < instances_confidence_threshold:
            break
        mask = instance_masks[inst_id]  # H,W
        mask_area = mask.sum().item()

        if mask_area == 0:
            continue

        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum().item()

        if intersect_area * 1.0 / mask_area > overlap_threshold:
            continue

        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": True,
                "category_id": instance_results.pred_classes[inst_id].item(),
            }
        )

    # Add semantic results to remaining empty areas
    semantic_labels = torch.unique(semantic_results)
    for semantic_label in semantic_labels:
        if semantic_label == 0:  # 0 is a special "thing" class
            continue
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        if mask.sum() < stuff_area_limit:
            continue

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {"id": current_segment_id, "isthing": False, "category_id": semantic_label.item()}
        )

    return panoptic_seg, segments_info

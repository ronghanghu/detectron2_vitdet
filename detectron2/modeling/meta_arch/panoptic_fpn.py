# -*- coding: utf-8 -*-

import torch
from torch import nn

from detectron2.structures import ImageList

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess, sem_seg_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from .semantic_seg import build_sem_seg_head

__all__ = ["PanopticFPN"]


@META_ARCH_REGISTRY.register()
class PanopticFPN(nn.Module):
    """
    Main class for Panoptic FPN architectures (see https://arxiv.org/abd/1901.02446).
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # weights for all losses
        self.semantic_loss_scale = cfg.MODEL.PANOPTIC_FPN.SEMANTIC_LOSS_SCALE
        self.instance_loss_scale = cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_SCALE
        self.rpn_loss_scale = cfg.MODEL.PANOPTIC_FPN.RPN_LOSS_SCALE

        # options when combining instance & semantic outputs
        self.combine_on = cfg.MODEL.PANOPTIC_FPN.COMBINE_ON
        self.combine_overlap_threshold = cfg.MODEL.PANOPTIC_FPN.COMBINE_OVERLAP_THRESHOLD
        self.combine_stuff_area_limit = cfg.MODEL.PANOPTIC_FPN.COMBINE_STUFF_AREA_LIMIT
        self.combine_instances_confidence_threshold = (
            cfg.MODEL.PANOPTIC_FPN.COMBINE_INSTANCES_CONFIDENCE_THRESHOLD
        )

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            targets: Instances
            sem_seg_gt: semantic segmentation ground truth.
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]: each dict is the results for one image. The dict
                contains the following keys:
                "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                "panoptic_seg": available when `PANOPTIC_FPN.COMBINE_ON`.
                    See the return value of
                    :func:`combine_semantic_and_instance_outputs` for its format.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "proposals" in batched_inputs[0]:
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        if "sem_seg_gt" in batched_inputs[0]:
            sem_seg_targets = [x["sem_seg_gt"].to(self.device) for x in batched_inputs]
            sem_seg_targets = ImageList.from_tensors(
                sem_seg_targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            sem_seg_targets = None
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, sem_seg_targets)

        if "targets" in batched_inputs[0]:
            detector_targets = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            detector_targets = None
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, detector_targets)
        detector_results, detector_losses = self.roi_heads(
            images, features, proposals, detector_targets
        )

        if self.training:
            losses = {}
            losses.update({k: v * self.semantic_loss_scale for k, v in sem_seg_losses.items()})
            losses.update({k: v * self.instance_loss_scale for k, v in detector_losses.items()})
            losses.update({k: v * self.rpn_loss_scale for k, v in proposal_losses.items()})
            return losses

        processed_results = []
        for sem_seg_result, detector_result, input_per_image, image_size in zip(
            sem_seg_results, detector_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            detector_r = detector_postprocess(detector_result, height, width)

            processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

            if self.combine_on:
                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r,
                    self.combine_overlap_threshold,
                    self.combine_stuff_area_limit,
                    self.combine_instances_confidence_threshold,
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
        return processed_results


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

    instance_masks = instance_results.pred_masks.to(dtype=torch.bool, device=panoptic_seg.device)

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

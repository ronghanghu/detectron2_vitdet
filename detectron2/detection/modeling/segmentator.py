import torch
from torch import nn

from detectron2.structures import ImageList

from .backbone import build_backbone
from .model_builder import META_ARCH_REGISTRY
from .postprocessing import (
    combine_semantic_and_instance_outputs,
    detector_postprocess,
    sem_seg_postprocess,
)
from .proposal_generator import build_proposal_generator
from .roi_heads.roi_heads import build_roi_heads
from .sem_seg_heads import build_sem_seg_head


@META_ARCH_REGISTRY.register()
class SemanticSegmentator(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    def __init__(self, cfg):
        super(SemanticSegmentator, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg)

        pixel_mean = torch.Tensor(cfg.INPUT.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.INPUT.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            sem_seg_gt: semantic segmentation ground truth
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "sem_seg" whose value is a
                Tensor of the output resolution that represents the
                per-pixel segmentation prediction.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)

        if "sem_seg_gt" in batched_inputs[0]:
            targets = [x["sem_seg_gt"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            targets = None
        results, losses = self.sem_seg_head(features, targets)

        if self.training:
            return losses

        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class PanopticFPN(nn.Module):
    """
    Main class for Panoptic FPN architectures (see https://arxiv.org/abd/1901.02446).
    """

    def __init__(self, cfg):
        super(PanopticFPN, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # weights for all losses
        self.semantic_loss_scale = cfg.MODEL.PANOPTIC.SEMANTIC_LOSS_SCALE
        self.instance_loss_scale = cfg.MODEL.PANOPTIC.INSTANCE_LOSS_SCALE
        self.rpn_loss_scale = cfg.MODEL.PANOPTIC.RPN_LOSS_SCALE

        # options when combining instance & semantic outputs
        self.combine_on = cfg.MODEL.PANOPTIC.COMBINE_ON
        self.combine_overlap_threshold = cfg.MODEL.PANOPTIC.COMBINE_OVERLAP_THRESHOLD
        self.combine_stuff_area_limit = cfg.MODEL.PANOPTIC.COMBINE_STUFF_AREA_LIMIT
        self.combine_instances_confidence_threshold = (
            cfg.MODEL.PANOPTIC.COMBINE_INSTANCES_CONFIDENCE_THRESHOLD
        )

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg)

        pixel_mean = torch.Tensor(cfg.INPUT.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.INPUT.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            targets: Instances
            sem_seg_gt: semantic segmentation ground truth.
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
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

            processed_results.append({"sem_seg": sem_seg_r, "detector": detector_r})  # TODO rename

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

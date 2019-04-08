import torch
from torch import nn

from detectron2.structures import ImageList

from .backbone import build_backbone
from .model_builder import META_ARCH_REGISTRY
from .postprocessing import detector_postprocess, sem_seg_postprocess
from .roi_heads.roi_heads import build_roi_heads
from .rpn.rpn import build_rpn
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
        """
        images = ImageList.from_list_of_dicts_by_image_key(
            batched_inputs, "image", self.backbone.size_divisibility
        ).to(self.device)
        features = self.backbone(images.tensor)

        if "sem_seg_gt" in batched_inputs[0]:
            targets = (
                ImageList.from_list_of_dicts_by_image_key(
                    batched_inputs,
                    "sem_seg_gt",
                    self.backbone.size_divisibility,
                    self.sem_seg_head.ignore_value,
                )
                .to(self.device)
                .tensor
            )
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
            processed_results.append(r)
        return processed_results


@META_ARCH_REGISTRY.register()
class PanopticFPN(nn.Module):
    """
    Main class for Panoptic FPN architectures (see https://arxiv.org/abd/1901.02446).
    """

    def __init__(self, cfg):
        super(PanopticFPN, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.semantic_loss_scale = cfg.MODEL.PANOPTIC.SEMANTIC_LOSS_SCALE
        self.instance_loss_scale = cfg.MODEL.PANOPTIC.INSTANCE_LOSS_SCALE
        self.rpn_loss_scale = cfg.MODEL.PANOPTIC.RPN_LOSS_SCALE

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg)

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
        images = ImageList.from_list_of_dicts_by_image_key(
            batched_inputs, "image", self.backbone.size_divisibility
        ).to(self.device)
        features = self.backbone(images.tensor)

        if "sem_seg_gt" in batched_inputs[0]:
            sem_seg_targets = (
                ImageList.from_list_of_dicts_by_image_key(
                    batched_inputs,
                    "sem_seg_gt",
                    self.backbone.size_divisibility,
                    self.sem_seg_head.ignore_value,
                )
                .to(self.device)
                .tensor
            )
        else:
            sem_seg_targets = None
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, sem_seg_targets)

        if "targets" in batched_inputs[0]:
            detector_targets = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            detector_targets = None
        proposals, proposal_losses = self.rpn(images, features, detector_targets)
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
            processed_results.append({"sem_seg": sem_seg_r, "detector": detector_r})
        return processed_results

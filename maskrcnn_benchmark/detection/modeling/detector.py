import torch
from torch import nn

from maskrcnn_benchmark.utils.registry import Registry

from .backbone import build_backbone
from .roi_heads.roi_heads import build_roi_heads
from .rpn.rpn import build_rpn


META_ARCH_REGISTRY = Registry("META_ARCH")


def build_detection_model(cfg):
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    return META_ARCH_REGISTRY.get(meta_arch)(cfg)


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Supports boxes, masks and keypoints
    This is very similar to what we had before, the difference being that now
    we construct the modules in __init__, instead of passing them as arguments
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)

        self.to(self.device)

    def forward(self, data):
        """
        Arguments:
            data: a tuple, produced by :class:`DetectionBatchCollator`.

        For now, the data contains images, targets, and dataset_dict
        images: ImageList
        targets: list[Instances]
        dataset_dict: other information that's not useful in training
        """

        images, targets, _ = data
        images = images.to(self.device)
        if targets is not None:
            targets = [t.to(self.device) for t in targets]

        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads.
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

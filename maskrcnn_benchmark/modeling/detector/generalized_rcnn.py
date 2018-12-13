"""
Proposal implementation for the model builder
Everything that constructs something is a function that is
or can be accessed by a string
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from ..backbone import build_backbone
from ..roi_heads.roi_heads import build_roi_heads
from ..rpn.rpn import build_rpn


class GeneralizedRCNN(nn.Module):
    """
    Main class for generalized r-cnn. Supports boxes, masks and keypoints
    This is very similar to what we had before, the difference being that now
    we construct the modules in __init__, instead of passing them as arguments
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.size_divisible = cfg.DATALOADER.SIZE_DIVISIBILITY
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)

        self.to(self.device)

    def preprocess(self, roidbs):
        """
        Returns:
            images: ImageList
            targets: list[BoxList]. None when not training.
        """
        images = [torch.as_tensor(x["image"].transpose(2, 0, 1).astype("float32")) for x in roidbs]
        images = to_image_list(images, self.size_divisible).to(self.device)

        if not self.training:
            return images, None

        targets = []
        for roidb in roidbs:
            image = roidb["image"]
            image_size = image.shape[1], image.shape[0]

            annos = roidb["annotations"]
            boxes = [obj["bbox"] for obj in annos]
            boxes = torch.as_tensor(boxes).reshape(-1, 4)
            target = BoxList(boxes, image_size, mode="xywh").convert("xyxy")

            classes = [obj["category_id"] for obj in annos]
            classes = torch.tensor(classes)
            target.add_field("labels", classes)

            masks = [obj["segmentation"] for obj in annos]
            masks = SegmentationMask(masks, image_size)
            target.add_field("masks", masks)
            target = target.clip_to_image(remove_empty=True)
            targets.append(target.to(self.device))
        return images, targets

    def forward(self, roidbs):
        """
        Arguments:
            roidbs (list): a list of training data. Each is a dict.
        """
        images, targets = self.preprocess(roidbs)

        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

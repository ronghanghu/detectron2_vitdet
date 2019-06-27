import torch
from torch import nn

import math
from detectron2.structures import ImageList

from detectron2.detection.modeling.backbone import build_backbone
from ..model_builder import META_ARCH_REGISTRY

__all__ = ["RetinaNet"]


@META_ARCH_REGISTRY.register()
class RetinaNet(nn.Module):
    """
    RetinaNet head. Creates FPN backbone, anchors and a head for classification
    and box regression. Calculates and applies smooth L1 loss and sigmoid focal
    loss.
    """

    def __init__(self, cfg):
        super(RetinaNet, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.box_cls_and_regression = RetinaNetClassifierRegressionHead(cfg)

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
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

         Returns:
             loss: placeholder loss for now. Will be replaced with smooth L1
                loss and sigmoid focal loss for box regression and classifier
                respectively.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)
        box_cls, box_regression = self.box_cls_and_regression(features)

        # TODO: Replace with actual loss.
        placeholder_loss = box_regression[0].mean()
        return {"placeholder_loss": placeholder_loss}


@META_ARCH_REGISTRY.register()
class RetinaNetClassifierRegressionHead(nn.Module):
    def __init__(self, cfg):
        """
        Creates a head for object classification subnet and box regression
        subnet. Although the two subnets share a common structure, they use
        separate parameters (unlike RPN).
        """
        super(RetinaNetClassifierRegressionHead, self).__init__()

        self.in_features = ["p3", "p4", "p5", "p6", "p7"]
        in_channels = cfg.MODEL.FPN.OUT_CHANNELS
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.RETINANET.ANCHOR_ASPECT_RATIOS) \
            * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE

        cls_subnet = []
        bbox_subnet = []
        for _ in range(cfg.MODEL.RETINANET.NUM_CONVS):
            cls_subnet.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_subnet.append(nn.ReLU())

        self.add_module('cls_subnet', nn.Sequential(*cls_subnet))
        self.add_module('bbox_subnet', nn.Sequential(*bbox_subnet))
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_logits,
                        self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (dict[str: Tensor]): mapping from feature map name to FPN
                feature map tensor in high to low resolution. Feature names
                follow the FPN paper convention "p<stage>", where stage has
                stride = 2 ** stage e.g., ["p3", "p3", ..., "p7"]. Each tensor
                in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): predicted the probability of object presence
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): predicted 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        features = [features[f] for f in self.in_features]
        for feature in features:
            logits.append(self.cls_logits(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg

import numpy as np
import borc.nn.weight_init as weight_init
import torch.nn as nn
from torch.nn import functional as F

from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry

SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")


def build_sem_seg_head(cfg):
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg)


def sem_seg_loss(pred_logits, targets, ignore_value):
    """
    Compute the semantic segmentation loss as per-pixel cross entropy ignoring pixels that have
    "ignore_value" in the target.

    Args:
        pred_logits (Tensor): A tensor of shape (B, C, H, W), where B is batch size,
            C is the number of classes, and H, W are the height and width of the predictions.
            The values are logits.
        targets (Tensor): A tensor of shape (B, H, W), where B is batch size,
            and H, W are the height and width of the predictions.
            The values are [0, ..., C-1] + [ignore_value].
    Returns:
        sem_seg_loss (Tensor): A scalar tensor containing the loss.
    """
    sem_seg_loss = F.cross_entropy(
        pred_logits, targets, reduction="mean", ignore_index=ignore_value
    )
    return sem_seg_loss


@SEM_SEG_HEADS_REGISTRY.register()
class SemSegHead(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg):
        super(SemSegHead, self).__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        feature_strides       = dict(cfg.MODEL.BACKBONE.COMPUTED_OUT_FEATURE_STRIDES)
        feature_channels      = dict(cfg.MODEL.BACKBONE.COMPUTED_OUT_FEATURE_CHANNELS)
        self.ignore_value     = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes           = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims             = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride    = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        norm                  = cfg.MODEL.SEM_SEG_HEAD.NORM
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features, targets=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        x = self.predictor(x)
        x = F.interpolate(x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)

        if self.training:
            losses = {}
            losses["loss_sem_seg"] = sem_seg_loss(x, targets, self.ignore_value)
            return [], losses
        else:
            return x, {}

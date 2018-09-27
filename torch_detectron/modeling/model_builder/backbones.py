from collections import OrderedDict

from torch import nn

from torch_detectron.modeling import fpn as fpn_module
from torch_detectron.modeling import resnet


def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    representation_size = cfg.MODEL.BACKBONE.OUT_CHANNELS
    fpn = fpn_module.FPN(
        layers=[256, 512, 1024, 2048],
        representation_size=representation_size,
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    return model


_BACKBONES = {"resnet": build_resnet_backbone, "resnet-fpn": build_resnet_fpn_backbone}


def build_backbone(cfg):
    # return _BACKBONES[cfg.MODEL.BACKBONE.CONV_TYPE](cfg)
    if "FPN" in cfg.MODEL.BACKBONE.CONV_BODY:
        return build_resnet_fpn_backbone(cfg)
    return build_resnet_backbone(cfg)

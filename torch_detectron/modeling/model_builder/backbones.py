from torch import nn

from torch_detectron.modeling import fpn
from torch_detectron.modeling import resnet


def build_resnet_backbone(cfg):
    return resnet.ResNet(cfg)


def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    representation_size = cfg.BACKBONE.OUT_CHANNELS
    fpn_top = fpn.FPN(
        layers=[256, 512, 1024, 2048],
        representation_size=representation_size,
        top_blocks=fpn.LastLevelMaxPool(),
    )
    model = nn.Sequential(body, fpn_top)
    return model


_BACKBONES = {"resnet": build_resnet_backbone, "resnet-fpn": build_resnet_fpn_backbone}


def build_backbone(cfg):
    # return _BACKBONES[cfg.BACKBONE.CONV_TYPE](cfg)
    if "FPN" in cfg.BACKBONE.CONV_BODY:
        return build_resnet_fpn_backbone(cfg)
    return build_resnet_backbone(cfg)

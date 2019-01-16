from collections import OrderedDict

from torch import nn

from . import fpn as fpn_module, resnet


def build_resnet_backbone(cfg):
    body = resnet.make_resnet_backbone(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model, [body.feature_strides[f] for f in cfg.MODEL.RESNETS.RETURN_FEATURES]


def build_resnet_fpn_backbone(cfg):
    body = resnet.make_resnet_backbone(cfg)
    body_strides = [body.feature_strides[f] for f in cfg.MODEL.RESNETS.RETURN_FEATURES]
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

    # Here FPN is (for now) hard-coded to have 4 feature map inputs.
    assert len(body_strides) == 4, len(body_strides)
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    strides = fpn.compute_feature_strides(body_strides)
    return model, strides


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.NAME == "ResNet"
    if cfg.MODEL.BACKBONE.USE_FPN:
        backbone, strides = build_resnet_fpn_backbone(cfg)
    else:
        backbone, strides = build_resnet_backbone(cfg)

    cfg.defrost()
    cfg.MODEL.BACKBONE.FEATURE_STRIDES = tuple(strides)
    if cfg.MODEL.BACKBONE.USE_FPN:
        # If > 0, this enforces that each collated batch should have a size divisible by SIZE_DIVISIBILITY
        # Image shape has to be divisible up to the last feature map extracted from the backbone
        # Otherwise FPN cannot add two featuremaps together.
        # This value is typically 32.
        cfg.DATALOADER.SIZE_DIVISIBILITY = strides[len(cfg.MODEL.RESNETS.RETURN_FEATURES) - 1]
    else:
        cfg.DATALOADER.SIZE_DIVISIBILITY = 0
    cfg.freeze()

    return backbone

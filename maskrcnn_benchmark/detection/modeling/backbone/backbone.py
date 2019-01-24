from maskrcnn_benchmark.layers import Backbone

from . import fpn as fpn_module, resnet


def build_resnet_fpn_backbone(cfg):
    """
    Args:
        cfg (yacs.CfgNode)

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = resnet.make_resnet_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = fpn_module.FPN(
        bottom_up=bottom_up, in_features=in_features, out_channels=out_channels
    )
    return backbone


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.NAME == "ResNet"
    if cfg.MODEL.BACKBONE.USE_FPN:
        backbone = build_resnet_fpn_backbone(cfg)
    else:
        backbone = resnet.make_resnet_backbone(cfg)
    assert isinstance(backbone, Backbone)

    # TODO: find a better way then the defrost/mutate cfg/freeze pattern used here
    cfg.defrost()
    # Hack: seprate keys (FEATURE_NAMES) and values because we cannot have a dict in yacs
    cfg.MODEL.BACKBONE.FEATURE_NAMES = tuple(backbone.out_features)
    cfg.MODEL.BACKBONE.FEATURE_STRIDES = tuple(backbone.out_feature_strides.values())
    cfg.MODEL.BACKBONE.FEATURE_CHANNELS = tuple(backbone.out_feature_channels.values())
    # If > 0, this enforces that each collated batch should have a size divisible by SIZE_DIVISIBILITY
    # Image shape has to be divisible up to the last feature map extracted from the backbone
    # Otherwise FPN cannot add two featuremaps together.
    # This value is typically 32.
    cfg.DATALOADER.SIZE_DIVISIBILITY = backbone.size_divisibility
    cfg.freeze()

    return backbone

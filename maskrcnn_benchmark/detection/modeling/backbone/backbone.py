from collections import OrderedDict
from torch import nn

from . import fpn as fpn_module, resnet


def build_resnet_backbone(cfg):
    """
    Args:
        cfg (yacs.CfgNode)

    Returns:
        model (torch.nn.Module): backbone module
        feature strides (dict[str: int]): mapping from named feature map to the
            stride of that feature map; includes only feature maps that are returned
            when calling `forward` on the backbone.
        feature channels (dict[str: int]): mapping from named feature map to the
            number of channels in that feature map; includes only feature maps
            that are returned when calling `forward` on the backbone.
        input size divisibility (int): non-negative integer containing input image
            size (height and width) divisibility requirements.
    """
    body = resnet.make_resnet_backbone(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    feature_strides = {f: body.feature_strides[f] for f in body.return_features}
    feature_channels = {f: body.feature_channels[f] for f in body.return_features}
    return model, feature_strides, feature_channels, body.size_divisibility


def build_resnet_fpn_backbone(cfg):
    """
    See: `build_resnet_backbone`.
    """
    body = resnet.make_resnet_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    body_strides = [body.feature_strides[f] for f in in_features]
    body_channels = [body.feature_channels[f] for f in in_features]
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_features=in_features,
        in_strides=body_strides,
        in_channels_list=body_channels,
        out_channels=out_channels,
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    return model, fpn.feature_strides, fpn.feature_channels, fpn.size_divisibility


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.NAME == "ResNet"
    if cfg.MODEL.BACKBONE.USE_FPN:
        backbone, strides, channels, size_divisibility = build_resnet_fpn_backbone(cfg)
    else:
        backbone, strides, channels, size_divisibility = build_resnet_backbone(cfg)

    # TODO: find a better way then the defrost/mutate cfg/freeze pattern used here
    cfg.defrost()
    # Hack: seprate keys (FEATURE_NAMES) and values because we cannot have a dict in yacs
    cfg.MODEL.BACKBONE.FEATURE_NAMES = tuple(strides.keys())
    cfg.MODEL.BACKBONE.FEATURE_STRIDES = tuple(strides.values())
    cfg.MODEL.BACKBONE.FEATURE_CHANNELS = tuple(channels.values())
    # If > 0, this enforces that each collated batch should have a size divisible by SIZE_DIVISIBILITY
    # Image shape has to be divisible up to the last feature map extracted from the backbone
    # Otherwise FPN cannot add two featuremaps together.
    # This value is typically 32.
    cfg.DATALOADER.SIZE_DIVISIBILITY = size_divisibility
    cfg.freeze()

    return backbone

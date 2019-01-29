from maskrcnn_benchmark.layers import Backbone

from . import fpn, resnet


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.NAME == "ResNet"
    if cfg.MODEL.BACKBONE.USE_FPN:
        backbone = fpn.build_resnet_fpn_backbone(cfg)
    else:
        backbone = resnet.build_resnet_backbone(cfg)
    assert isinstance(backbone, Backbone)

    # TODO: find a better way then the defrost/mutate cfg/freeze pattern used here
    cfg.defrost()
    # Convert dicts to immutable ((key1, val1), (key2, val2), ...) tuples
    cfg.MODEL.BACKBONE.OUT_FEATURE_STRIDES = tuple(backbone.out_feature_strides.items())
    cfg.MODEL.BACKBONE.OUT_FEATURE_CHANNELS = tuple(backbone.out_feature_channels.items())
    # If > 0, this enforces that each collated batch should have a size divisible by SIZE_DIVISIBILITY
    # Image shape has to be divisible up to the last feature map extracted from the backbone
    # Otherwise FPN cannot add two featuremaps together.
    # This value is typically 32.
    cfg.DATALOADER.SIZE_DIVISIBILITY = backbone.size_divisibility
    cfg.freeze()

    return backbone
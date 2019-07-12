from detectron2.utils.registry import Registry

from .backbone import Backbone

BACKBONE_REGISTRY = Registry("BACKBONE")


def build_backbone(cfg):
    """
    Returns:
        an instance of :class:`Backbone`
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)

    assert isinstance(backbone, Backbone)

    # Convert dicts to immutable ((key1, val1), (key2, val2), ...) tuples
    cfg.MODEL.BACKBONE.COMPUTED_OUT_FEATURE_STRIDES = tuple(backbone.out_feature_strides.items())
    cfg.MODEL.BACKBONE.COMPUTED_OUT_FEATURE_CHANNELS = tuple(backbone.out_feature_channels.items())

    return backbone

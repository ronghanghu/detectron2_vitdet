from maskrcnn_benchmark.utils.registry import Registry
BACKBONE_REGISTRY = Registry("BACKBONE")

from .build import build_backbone  # noqa

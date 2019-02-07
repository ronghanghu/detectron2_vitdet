from maskrcnn_benchmark.utils.registry import Registry


# BACKBONE_REGISTRY must be defined before importing build_backbone
BACKBONE_REGISTRY = Registry("BACKBONE")
from .build import build_backbone  # noqa F401 isort:skip

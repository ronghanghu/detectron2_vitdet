# flake8: noqa

from .detector import build_detection_model, META_ARCH_REGISTRY
from .roi_heads.roi_heads import build_roi_heads, ROIHeads, ROI_HEADS_REGISTRY

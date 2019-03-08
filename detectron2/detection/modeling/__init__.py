# flake8: noqa

from .backbone import BACKBONE_REGISTRY
from .detector import META_ARCH_REGISTRY, build_detection_model
from .roi_heads.box_head import ROI_BOX_HEAD_REGISTRY, build_box_head
from .roi_heads.keypoint_head import ROI_KEYPOINT_HEAD_REGISTRY, build_keypoint_head
from .roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY, build_mask_head
from .roi_heads.roi_heads import ROI_HEADS_REGISTRY, ROIHeads, build_roi_heads

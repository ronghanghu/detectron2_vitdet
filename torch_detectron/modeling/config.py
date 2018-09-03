from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.MIN_SIZE_TRAIN = 800#(800,)
_C.INPUT.MAX_SIZE_TRAIN = 1333
_C.INPUT.MIN_SIZE_TEST = 800
_C.INPUT.MAX_SIZE_TEST = 1333
_C.INPUT.SIZE_DIVISIBILITY = 0
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
_C.INPUT.PIXEL_STD = [1., 1., 1.]


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.SIZE_DIVISIBILITY = 0
_C.DATALOADER.IMAGES_PER_BATCH_TRAIN = 2
_C.DATALOADER.IMAGES_PER_BATCH_TEST = 1

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.BACKBONE.CONV_BODY = "R-50-C4"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.BACKBONE.FREEZE_CONV_BODY_AT = 2
_C.BACKBONE.OUTPUT_DIM = 256 * 4


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.USE_FPN = False
_C.MODEL.RPN.SCALES = (0.125, 0.25, 0.5, 1., 2.)
_C.MODEL.RPN.BASE_ANCHOR_SIZE = 256
_C.MODEL.RPN.ANCHOR_STRIDE = 16
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
_C.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# TODO separate in train and test
_C.MODEL.RPN.PRE_NMS_TOP_N = 12000
_C.MODEL.RPN.POST_NMS_TOP_N = 2000
# TODO this is not as nice
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
_C.MODEL.RPN.NMS_THRESH = 0.7
_C.MODEL.RPN.MIN_SIZE = 0
# _C.MODEL.RPN.WEIGHTS = None
_C.MODEL.RPN.FPN_POST_NMS_TOP_N = 2000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000


# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.USE_FPN = False
# _C.MODEL.ROI_HEADS.POOLER.DROP_LAST = False
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

#_C.MODEL.ROI_HEADS.HEAD_BUILDER = None
#_C.MODEL.ROI_HEADS.MASK_BUILDER = None
_C.MODEL.ROI_HEADS.MASK_RESOLUTION = 14
#_C.MODEL.ROI_HEADS.MASK_POOLER = None

# Only used on test mode
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
_C.MODEL.ROI_HEADS.NMS = 0.5
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100


_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81

_C.MODEL.SHARE_FEATURES_DURING_TRAINING = True


# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_C.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
_C.RESNETS.RES5_DILATION = 1



# ---------------------------------------------------------------------------- #
# Exporting the config
# ---------------------------------------------------------------------------- #
cfg = _C

"""
Base configuration file for detection models.
As this is a python file, it suports arbitrary data structures.

The configuration file represents a tree structure, and the
user can modify arbitrary nodes of the tree such that we
can obtain a lot of flexibility.

For example, if the model structure that is defined here
is not generic enough for a particular use-case,
the user can simply create a dummy class ModelBuilder
that returns the model uppon __call__, and add it to
config.MODEL.

For example:
class MyModelBuilder(ConfigClass):
    def __call__(self):
        param1 = self.PARAM1
        return MyBrandNewModel(param1)

config.MODEL = MyModelBuilder()
config.MODEL.PARAM1 = 5

The only required fields for the moment are:
    - MODEL: should be a callable that returns the model
    - TRAIN.DATA: should be a callable that returns a DataLoader
    - TEST.DATA: same as TRAIN.DATA
    - SOLVER.OPTIM: callable that takes the result of MODEL() and
        returns the optimizer
    - SOLVER.SCHEDULER: callable that takes the result of SOLVER.OPTIM()
        and return the learning rate scheduler

Everything in between can be modified / replaced.
"""
import copy

import torch

import torch_detectron.helpers.data as _data
import torch_detectron.helpers.model as _model
import torch_detectron.helpers.solver as _solver
from torch_detectron.helpers.config_utils import AttrDict
from torch_detectron.helpers.config_utils import link_nodes
from torch_detectron.layers import ROIAlign as _ROIAlign

_C = AttrDict()

_C.DEVICE = torch.device("cuda")
_C.DO_TEST = True
# if None, doesn't save checkpoints
_C.SAVE_DIR = None
# if None, doesn't load from checkpoint
_C.CHECKPOINT = None

# ============================================================== #

# TODO it makes sense to have a TRAIN / TEST different model maybe
# but this makes loading checkpoints harder

_C.MODEL = _model.ModelBuilder()
_C.MODEL.RPN_ONLY = False
_C.MODEL.USE_MASK = False
_C.MODEL._INTERNAL_REPRESENTATION_SIZE = 256 * 4

_C.MODEL.BACKBONE = _model.BackboneBuilder()
_C.MODEL.BACKBONE.WEIGHTS = None
_C.MODEL.BACKBONE.BUILDER = _model.resnet50_conv4_body

RPN = _model.RPNBuilder()
RPN.USE_FPN = False
RPN.SCALES = (0.125, 0.25, 0.5, 1., 2.)
RPN.BASE_ANCHOR_SIZE = 256
RPN.ANCHOR_STRIDE = 16
RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
RPN.STRADDLE_THRESH = 0
RPN.MATCHED_THRESHOLD = 0.7
RPN.UNMATCHED_THRESHOLD = 0.3
RPN.BATCH_SIZE_PER_IMAGE = 256
RPN.POSITIVE_FRACTION = 0.5
# TODO separate in train and test
RPN.PRE_NMS_TOP_N = 12000
RPN.POST_NMS_TOP_N = 2000
# TODO this is not as nice
RPN.PRE_NMS_TOP_N_TEST = 6000
RPN.POST_NMS_TOP_N_TEST = 1000
RPN.NMS_THRESH = 0.7
RPN.MIN_SIZE = 0
RPN.WEIGHTS = None
RPN.FPN_POST_NMS_TOP_N = 2000
RPN.FPN_POST_NMS_TOP_N_TEST = 2000

POOLER = _model.PoolerBuilder()
# TODO decide if we want to decompose in elementary objects
POOLER.MODULE = _ROIAlign((14, 14), 1.0 / 16, 0)

HEADS = _model.DetectionAndMaskHeadsBuilder()
HEADS.USE_FPN = False
HEADS.POOLER = POOLER
HEADS.MATCHED_THRESHOLD = 0.5
HEADS.UNMATCHED_THRESHOLD = 0.0
HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
HEADS.BATCH_SIZE_PER_IMAGE = 512
HEADS.POSITIVE_FRACTION = 0.25
HEADS.NUM_CLASSES = 81
HEADS.WEIGHTS = None
HEADS.BUILDER = _model.resnet50_conv5_head

HEADS.HEAD_BUILDER = None
HEADS.MASK_BUILDER = None
HEADS.MASK_RESOLUTION = 14
HEADS.MASK_POOLER = None

# Only used on test mode
HEADS.SCORE_THRESH = 0.05
HEADS.NMS = 0.5
HEADS.DETECTIONS_PER_IMG = 100

_C.MODEL.REGION_PROPOSAL = RPN
_C.MODEL.HEADS = HEADS


# create links between nodes
link_nodes(_C, 'MODEL.REGION_PROPOSAL.RPN_ONLY', 'MODEL.RPN_ONLY')
link_nodes(_C, 'MODEL.HEADS.USE_MASK', 'MODEL.USE_MASK')

link_nodes(_C, 'MODEL.BACKBONE.REPRESENTATION_SIZE', 'MODEL._INTERNAL_REPRESENTATION_SIZE')
link_nodes(_C, 'MODEL.REGION_PROPOSAL.NUM_INPUT_FEATURES', 'MODEL._INTERNAL_REPRESENTATION_SIZE')
link_nodes(_C, 'MODEL.HEADS.REPRESENTATION_SIZE', 'MODEL._INTERNAL_REPRESENTATION_SIZE')

# ============================================================== #

_C.SOLVER = AttrDict()
_C.SOLVER.MAX_ITER = 40000
_C.SOLVER.OPTIM = _solver._SGDOptimizer()
_C.SOLVER.OPTIM.BASE_LR = 0.001
_C.SOLVER.OPTIM.MOMENTUM = 0.9
_C.SOLVER.OPTIM.WEIGHT_DECAY = 0.0005
_C.SOLVER.OPTIM.BASE_LR_BIAS = 0.002
_C.SOLVER.OPTIM.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.SCHEDULER = _solver._WarmupMultiStepLR()
_C.SOLVER.SCHEDULER.STEPS = [30000]
_C.SOLVER.SCHEDULER.GAMMA = 0.1
_C.SOLVER.SCHEDULER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.SCHEDULER.WARMUP_ITERS = 500
_C.SOLVER.SCHEDULER.WARMUP_METHOD = "linear"

# ============================================================== #

_C.TRAIN = AttrDict()

_C.TRAIN.DATA = _data._Data()
_C.TRAIN.DATA.DATASET = _data._COCODataset()
_C.TRAIN.DATA.DATASET.FILES = ()

_C.TRAIN.DATA.DATALOADER = _data._DataLoader()
_C.TRAIN.DATA.DATALOADER.NUM_WORKERS = 4
_C.TRAIN.DATA.DATALOADER.IMAGES_PER_BATCH = 2

_C.TRAIN.DATA.DATALOADER.SAMPLER = _data._DataSampler()
_C.TRAIN.DATA.DATALOADER.SAMPLER.SHUFFLE = True
_C.TRAIN.DATA.DATALOADER.SAMPLER.DISTRIBUTED = False

_C.TRAIN.DATA.DATALOADER.COLLATOR = _data._Collator()
_C.TRAIN.DATA.DATALOADER.COLLATOR.SIZE_DIVISIBLE = 0

_C.TRAIN.DATA.TRANSFORM = _data._DataTransform()
_C.TRAIN.DATA.TRANSFORM.MIN_SIZE = 800
_C.TRAIN.DATA.TRANSFORM.MAX_SIZE = 1333
_C.TRAIN.DATA.TRANSFORM.MEAN = [102.9801, 115.9465, 122.7717]
_C.TRAIN.DATA.TRANSFORM.STD = [1, 1, 1]
_C.TRAIN.DATA.TRANSFORM.TO_BGR255 = True
_C.TRAIN.DATA.TRANSFORM.FLIP_PROB = 0.5

# ============================================================== #
_C.TEST = AttrDict()

_C.TEST.DATA = _data._Data()
_C.TEST.DATA.DATASET = _data._COCODataset()
_C.TEST.DATA.DATASET.FILES = ()

_C.TEST.DATA.DATALOADER = _data._DataLoader()
_C.TEST.DATA.DATALOADER.NUM_WORKERS = 4
_C.TEST.DATA.DATALOADER.IMAGES_PER_BATCH = 1

_C.TEST.DATA.DATALOADER.SAMPLER = _data._DataSampler()
_C.TEST.DATA.DATALOADER.SAMPLER.SHUFFLE = False
_C.TEST.DATA.DATALOADER.SAMPLER.DISTRIBUTED = False

_C.TEST.DATA.DATALOADER.COLLATOR = _data._Collator()
_C.TEST.DATA.DATALOADER.COLLATOR.SIZE_DIVISIBLE = 0

_C.TEST.DATA.TRANSFORM = _data._DataTransform()
_C.TEST.DATA.TRANSFORM.MIN_SIZE = 800
_C.TEST.DATA.TRANSFORM.MAX_SIZE = 1333
_C.TEST.DATA.TRANSFORM.MEAN = [102.9801, 115.9465, 122.7717]
_C.TEST.DATA.TRANSFORM.STD = [1, 1, 1]
_C.TEST.DATA.TRANSFORM.TO_BGR255 = True
_C.TEST.DATA.TRANSFORM.FLIP_PROB = 0


def get_default_config():
    """Return a config object populated with defaults."""
    return copy.deepcopy(_C)

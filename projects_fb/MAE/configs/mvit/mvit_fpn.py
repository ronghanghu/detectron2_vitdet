from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2 import model_zoo

from .mvit_config import config as cfg
from ...mvit.mvit import MViT


cfg.MVIT.PATCH_2D = True
cfg.MVIT.MODE = "conv"
cfg.MVIT.CLS_EMBED_ON = False
cfg.MVIT.PATCH_KERNEL = [7, 7]
cfg.MVIT.PATCH_STRIDE = [4, 4]
cfg.MVIT.PATCH_PADDING = [3, 3]
cfg.MVIT.DROPPATH_RATE = 0.2
cfg.MVIT.DEPTH = 16
cfg.MVIT.DIM_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
cfg.MVIT.HEAD_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
cfg.MVIT.POOL_KVQ_KERNEL = [1, 3, 3]
cfg.MVIT.POOL_KV_STRIDE = [[0, 1, 4, 4], [1, 1, 2, 2], [2, 1, 2, 2], [3, 1, 1, 1], [4, 1, 1, 1], [5, 1, 1, 1], [6, 1, 1, 1], [7, 1, 1, 1], [8, 1, 1, 1], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1], [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 1, 1], [15, 1, 1, 1]]
cfg.MVIT.POOL_Q_STRIDE = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
cfg.MVIT.OUT_FEATURES = ["scale2", "scale3", "scale4", "scale5"]

model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
model.pixel_mean = [123.675, 116.28, 103.53]
model.pixel_std = [58.395, 57.12, 57.375]
model.input_format = "RGB"
model.backbone.bottom_up = L(MViT)(cfg=cfg, in_chans=3)
model.backbone.in_features = ["scale2", "scale3", "scale4", "scale5"]

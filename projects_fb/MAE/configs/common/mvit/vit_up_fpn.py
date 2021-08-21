from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2 import model_zoo

from .mvit_config import config as cfg
from ...mvit.mvit import MViT, ViTUp


cfg.MVIT.PATCH_2D = True
cfg.MVIT.CLS_EMBED_ON = True
cfg.MVIT.PATCH_KERNEL = [16, 16]
cfg.MVIT.PATCH_STRIDE = [16, 16]
cfg.MVIT.PATCH_PADDING = [0, 0]
cfg.MVIT.EMBED_DIM = 768
cfg.MVIT.NUM_HEADS = 12
cfg.MVIT.MLP_RATIO = 4.0
cfg.MVIT.QKV_BIAS = True
cfg.MVIT.DROPPATH_RATE = 0.2
cfg.MVIT.DEPTH = 12
cfg.MVIT.DIM_MUL = []
cfg.MVIT.HEAD_MUL = []
cfg.MVIT.POOL_KV_STRIDE = []
cfg.MVIT.POOL_Q_STRIDE = []
cfg.MVIT.OUT_FEATURES = ["block2", "block5", "block8", "block11"]

model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
model.pixel_mean = [123.675, 116.28, 103.53]
model.pixel_std = [58.395, 57.12, 57.375]
model.input_format = "RGB"
model.backbone.bottom_up = L(ViTUp)(
    net=L(MViT)(cfg=cfg, in_chans=3),
    in_features=["block2", "block5", "block8", "block11"],
    scale_factors=(4.0, 2.0, 1.0, 0.5),
)
model.backbone.in_features = ["block2", "block5", "block8", "block11"]

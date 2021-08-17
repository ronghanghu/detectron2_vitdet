from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

from detectron2 import model_zoo
train = model_zoo.get_config("common/train.py").train

from ..mvit.coco import dataloader
from ..mvit.optim import AdamW as optimizer
from ..mvit.mvit_s5 import model


train.amp.enabled = True
train.ddp.fp16_compression = False
train.init_checkpoint = "/checkpoint/kaiminghe/converted/moco_pretrained_tf2pt.pth"

num_node = 8

dataloader.train.total_batch_size = 8 * num_node
optimizer.lr = 0.00002 * num_node

# 3x
total_steps = 3 * 90000 * 2  # bs = 8
train.max_iter = int(total_steps / num_node)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[
            int((total_steps - 120000) / num_node),
            int((total_steps - 40000) / num_node),
            int(total_steps / num_node),
        ],
    ),
    warmup_length=2000 / num_node / train.max_iter,
    warmup_factor=0.001,
)

model.backbone.cfg.MVIT.CLS_EMBED_ON = True
model.backbone.cfg.MVIT.PATCH_KERNEL = [16, 16]
model.backbone.cfg.MVIT.PATCH_STRIDE = [16, 16]
model.backbone.cfg.MVIT.PATCH_PADDING = [0, 0]
model.backbone.cfg.MVIT.EMBED_DIM = 768
model.backbone.cfg.MVIT.NUM_HEADS = 12
model.backbone.cfg.MVIT.MLP_RATIO = 4.0
model.backbone.cfg.MVIT.QKV_BIAS = True
model.backbone.cfg.MVIT.DROPPATH_RATE = 0.2
model.backbone.cfg.MVIT.DEPTH = 12
model.backbone.cfg.MVIT.DIM_MUL = []
model.backbone.cfg.MVIT.HEAD_MUL = []
model.backbone.cfg.MVIT.POOL_KV_STRIDE = []
model.backbone.cfg.MVIT.POOL_Q_STRIDE = []
model.backbone.cfg.MVIT.OUT_FEATURES = ["scale2"]

model.proposal_generator.in_features = ["scale2"]

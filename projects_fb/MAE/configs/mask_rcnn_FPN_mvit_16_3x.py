from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

from detectron2 import model_zoo
#optimizer = model_zoo.get_config("common/optim.py").SGD
#lr_multiplier = model_zoo.get_config("common/coco_schedule.py").lr_multiplier_3x
# dataloader = model_zoo.get_config("common/data/coco.py").dataloader
# model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
train = model_zoo.get_config("common/train.py").train

from .mvit.coco import dataloader
from .mvit.optim import AdamW as optimizer
from .mvit.mvit_fpn import model


train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "manifold://fair_logging/tree/lyttonhao/mvit_pretrain/f284447571_MVIT_B_16_CONV_nocls_8n_qkv.pth"
train.init_checkpoint = "manifold://fair_logging/tree/lyttonhao/2021-07-08-235320-488/checkpoints/checkpoint_epoch_00300.pyth"
train.init_checkpoint = ""

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

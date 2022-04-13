from ..mask_rcnn_vit_b_w14_4res_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer
)

model.backbone.bottom_up.net.residual_block_indexes = []
model.backbone.bottom_up.net.window_block_indexes = [
    0,
    1,  # 2  global
    3,
    4,  # 5  global
    6,
    7,  # 8  global
    9,
    10,  # 11 global
]

train.init_checkpoint = "Please specify"

model.backbone.bottom_up.net.pretrain_img_size = 224

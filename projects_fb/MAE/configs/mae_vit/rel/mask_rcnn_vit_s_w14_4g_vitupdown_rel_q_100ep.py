from ..mask_rcnn_vit_s_w14_4res_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer
)

from ....vit.vit import ViTUpDimDown

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

model.backbone.bottom_up.update(_target_=ViTUpDimDown)


# use rel
model.backbone.bottom_up.net.rel_q = True

from ..mask_rcnn_vit_l_w14_4res_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer
)

from ....vit.vit import ViTUpDimDown

model.backbone.bottom_up.net.residual_block_indexes = []
model.backbone.bottom_up.net.window_block_indexes = [
    0, 1, 2, 3, 4,       # 5 global
    6, 7, 8, 9, 10,      # 11 global
    12, 13, 14, 15, 16,  # 17 global
    18, 19, 20, 21, 22,  # 23 global
]

model.backbone.bottom_up.update(_target_=ViTUpDimDown)

# use rel
model.backbone.bottom_up.net.rel_q = True

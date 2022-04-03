from ..mask_rcnn_vit_h_16_w14_4res_allwd_50ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer
)

model.backbone.bottom_up.net.residual_block_indexes = []
model.backbone.bottom_up.net.window_block_indexes = [
    0, 1, 2, 3, 4, 5, 6,          # 7 global
    8, 9, 10, 11, 12, 13, 14,      # 15 global
    16, 17, 18, 19, 20, 21, 22,  # 23 global
    24, 25, 26, 27, 28, 29, 30,   # 31 global
]

from ....vit.vit import ViTUpDimDown

model.backbone.bottom_up.update(_target_=ViTUpDimDown)

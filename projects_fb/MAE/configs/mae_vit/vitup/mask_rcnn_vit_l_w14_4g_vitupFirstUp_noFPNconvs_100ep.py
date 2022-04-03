from ..mask_rcnn_vit_l_w14_4res_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer
)

from ....vit.vit import ViTUpFirstUp
from ....beit.fpn import FPNWoTopdown

model.backbone.bottom_up.net.residual_block_indexes = []
model.backbone.bottom_up.net.window_block_indexes = [
    0, 1, 2, 3, 4,       # 5 global
    6, 7, 8, 9, 10,      # 11 global
    12, 13, 14, 15, 16,  # 17 global
    18, 19, 20, 21, 22,  # 23 global
]

model.backbone.bottom_up.update(
    _target_=ViTUpFirstUp,
    out_features=["s1", "s2", "s3", "s4"],
    upscale=4.0,
    scale_factors=[1.0, 1 / 2.0, 1 / 4.0, 1 / 8.0],
    mode="max",
    out_dim=256,
    use_finest_conv=True,
    finest_3x3_conv_num=1,
)
model.backbone.in_features = "${.bottom_up.out_features}"

model.backbone.bottom_up.net.out_features = ["s1"]
model.backbone.bottom_up.net.out_block_indexes = [23]


model.backbone.update(
    _target_=FPNWoTopdown,
    use_lateral_conv=False,
    out_conv=False,
)

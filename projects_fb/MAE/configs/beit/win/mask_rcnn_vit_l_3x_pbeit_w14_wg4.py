from ..mask_rcnn_vit_l_3x_pbeit import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

model.backbone.window_size = 14
model.backbone.window_block_indexes = [
    0, 1, 2, 3, 4, 
    6, 7, 8, 9, 10, 
    12, 13, 14, 15, 16, 
    18, 19, 20, 21, 22,
]
model.backbone.checkpoint_block_num = 0
model.backbone.use_cls_token_det = False
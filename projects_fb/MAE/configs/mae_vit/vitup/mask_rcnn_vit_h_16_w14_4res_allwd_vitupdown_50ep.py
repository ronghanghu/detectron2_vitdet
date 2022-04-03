from ..mask_rcnn_vit_h_16_w14_4res_allwd_50ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer
)


from ....vit.vit import ViTUpDimDown

model.backbone.bottom_up.update(_target_=ViTUpDimDown)

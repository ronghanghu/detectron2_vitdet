from .cascade_mask_rcnn_vit_b_w14_4res_50ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer
)

model.backbone.bottom_up.net.k_bias = True

# from sup init (MAE-Large-sup, X% )
train.init_checkpoint = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"

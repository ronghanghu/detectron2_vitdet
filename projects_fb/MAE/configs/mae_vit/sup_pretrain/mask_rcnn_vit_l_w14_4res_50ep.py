from ..mask_rcnn_vit_l_w14_4res_50ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer
)

# from sup init (MAE-Large-sup, X% )
train.init_checkpoint = "/checkpoint/kaiminghe/logs/deit/scratch/210921-112325-4n-SupBeiT-ep200-lr1.6e-3-wd0.3-ema0.9999-abs-NOrelpos-NOlayerscale-NOerase-droppath0.2-batch4k-cls-elem-b0.95/checkpoint-199.pth"  

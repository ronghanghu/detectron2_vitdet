from ..vit.mask_rcnn_vit_b_3x import train, optimizer, lr_multiplier, dataloader

from detectron2.config import LazyCall as L
from detectron2 import model_zoo

from ...vit.vit import VisionTransformerDet, ViTUp1


model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
model.pixel_mean = [123.675, 116.28, 103.53]
model.pixel_std = [58.395, 57.12, 57.375]
model.input_format = "RGB"
model.backbone.bottom_up = L(ViTUp1)(
    net=L(VisionTransformerDet)(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.2,
        out_features=["block2", "block5", "block8", "block11"],
    ),
    in_features="${.net.out_features}",
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    embed_dim="${.net.embed_dim}",
    mode=5,
)
model.backbone.in_features = "${.bottom_up.net.out_features}"


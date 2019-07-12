from torch import nn

from detectron2.layers import BasicStem, BottleneckBlock, DeformBottleneckBlock, ResNet, make_stage

from . import BACKBONE_REGISTRY


def build_resnet_head(cfg):
    # For now, assumed to be a res5 head.
    # fmt: off
    stage_channel_factor    = 2 ** 3  # res5 is 8x res2
    num_groups              = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group         = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels     = num_groups * width_per_group * stage_channel_factor
    out_channels            = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
    stride_in_1x1           = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    norm                    = cfg.MODEL.RESNETS.NORM
    # fmt: on

    blocks = make_stage(
        BottleneckBlock,
        3,
        first_stride=2,
        in_channels=out_channels // 2,
        bottleneck_channels=bottleneck_channels,
        out_channels=out_channels,
        num_groups=num_groups,
        norm=norm,
        stride_in_1x1=stride_in_1x1,
    )
    return nn.Sequential(*blocks)


@BACKBONE_REGISTRY.register()
def build_resnet_backbone(cfg):
    # TODO registration of new blocks/stems
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=len(cfg.MODEL.PIXEL_MEAN),
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False

    # fmt: off
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "bottleneck_channels": bottleneck_channels,
            "out_channels": out_channels,
            "num_groups": num_groups,
            "norm": norm,
            "stride_in_1x1": stride_in_1x1,
            "dilation": dilation,
        }
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DeformBottleneckBlock
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = BottleneckBlock
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features)

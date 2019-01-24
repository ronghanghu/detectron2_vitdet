from torch import nn

from maskrcnn_benchmark.layers import BasicStem, BottleneckBlock, ResNet, make_stage


def make_resnet_head(cfg):
    # For now, assumed to be a res5 head.
    stage_channel_factor = 2 ** 3  # res5 is 8x res2
    resnet_cfg = cfg.MODEL.RESNETS
    num_groups = resnet_cfg.NUM_GROUPS
    width_per_group = resnet_cfg.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group * stage_channel_factor
    out_channels = resnet_cfg.RES2_OUT_CHANNELS * stage_channel_factor
    stride_in_1x1 = resnet_cfg.STRIDE_IN_1X1

    blocks = make_stage(
        BottleneckBlock,
        3,
        first_stride=2,
        in_channels=out_channels // 2,
        bottleneck_channels=bottleneck_channels,
        out_channels=out_channels,
        num_groups=num_groups,
        norm="FrozenBN",
        stride_in_1x1=stride_in_1x1,
    )
    return nn.Sequential(*blocks)


def make_resnet_backbone(cfg):
    # TODO registration of new blocks/stems
    resnet_cfg = cfg.MODEL.RESNETS
    stem = BasicStem(out_channels=resnet_cfg.STEM_OUT_CHANNELS, norm="FrozenBN")
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False

    out_features = resnet_cfg.OUT_FEATURES
    depth = resnet_cfg.DEPTH

    num_groups = resnet_cfg.NUM_GROUPS
    width_per_group = resnet_cfg.WIDTH_PER_GROUP
    in_channels = resnet_cfg.STEM_OUT_CHANNELS
    bottleneck_channels = num_groups * width_per_group
    out_channels = resnet_cfg.RES2_OUT_CHANNELS
    stride_in_1x1 = resnet_cfg.STRIDE_IN_1X1
    res5_dilation = resnet_cfg.RES5_DILATION
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
        blocks = make_stage(
            BottleneckBlock,
            num_blocks_per_stage[idx],
            first_stride=first_stride,
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm="FrozenBN",
            stride_in_1x1=stride_in_1x1,
            dilation=dilation,
        )
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features)

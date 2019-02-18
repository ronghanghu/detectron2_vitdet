import math
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import Backbone, Conv2d, weight_init

from . import BACKBONE_REGISTRY, resnet


class FPN(Backbone):
    """
    Module that adds FPN on top of a list of feature maps.
    """

    def __init__(self, bottom_up, in_features, out_channels, norm="", top_block=True):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): either "" or "GN".
            top_block (bool, optional): if provided, an extra 2x2 max pooling
                is added on the output of the last (lowest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        in_strides = [bottom_up.feature_strides[f] for f in in_features]
        in_channels = [bottom_up.feature_channels[f] for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""

        for idx, in_channels in enumerate(in_channels):
            lateral_norm = nn.GroupNorm(32, out_channels) if norm == "GN" else None
            output_norm = nn.GroupNorm(32, out_channels) if norm == "GN" else None

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(in_strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
        if self.top_block:
            self._feature_strides["p{}".format(stage + 1)] = 2 ** (stage + 1)
        self._out_features = list(self._feature_strides.keys())
        self._feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]

    def forward(self, x):
        """
        Args:
            input (dict[str: Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str: Tensor]: mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        x = self.bottom_up(x)
        x = [x[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            results.insert(0, output_conv(prev_features))

        if self.top_block:
            results.append(F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0))

        return dict(zip(self._out_features, results))


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Stides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone(cfg):
    """
    Args:
        cfg (yacs.CfgNode)

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = resnet.build_resnet_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
    )
    return backbone

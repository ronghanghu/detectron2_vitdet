import torch.nn.functional as F
from torch import nn

import maskrcnn_benchmark.layers.weight_init as weight_init


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    """

    def __init__(self, in_channels_list, out_channels, top_block=True):
        """
        Args:
            in_channels_list (list[int]): number of channels for each input feature
                map in `forward()`.
            out_channels (int): number of channels in the output feature maps.
            top_block (bool, optional): if provided, an extra 2x2 max pooling
                is added on the output of the last (lowest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        lateral_convs = []
        output_convs = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("fpn_lateral{}".format(idx), lateral_conv)
            self.add_module("fpn_output{}".format(idx), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block

    def forward(self, x):
        """
        Args:
            x (list[Tensor]): feature maps for each feature level in high to low
                resolution order.

        Returns:
            tuple[Tensor]: feature maps from FPN in high to low resolution order.
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        x = x[::-1]
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
            last_results = F.max_pool2d(results[-1], 1, 2, 0)
            results.append(last_results)

        return tuple(results)

    def compute_feature_strides(self, input_strides):
        """
        Args:
            input_strides (list[int]): a list of strides correspond to the
                input feature maps.

        Returns:
            list[int]: the stride for each output feature map.
        """
        for i, stride in enumerate(input_strides):
            if i > 0:
                assert stride == 2 * input_strides[i - 1]
        if self.top_block:
            return input_strides + [input_strides[-1] * 2]
        else:
            return input_strides

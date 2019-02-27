import numpy as np
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d, weight_init


class FastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, input_size, num_conv, conv_dim, num_fc, fc_dim, norm=""):
        """
        Args:
            input_size: channels, height, width
            num_conv: int
            conv_dim: int
            num_fc: int
            fc_dim: int
            norm (str): normalization for the conv layers
        """
        super(FastRCNNConvFCHead, self).__init__()
        assert isinstance(input_size, (list, tuple))
        assert len(input_size) == 3
        assert num_conv + num_fc > 0

        self._output_size = input_size

        self.conv_norm_relus = []
        for k in range(num_conv):
            norm_module = nn.GroupNorm(32, conv_dim) if norm == "GN" else None
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=norm_module,
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            x = x.view(x.shape[0], -1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_size(self):
        return self._output_size


# TODO: use registration
def build_box_head(cfg, input_size):
    """
    Args:
        input_size: channels, or (channels, height, width)
    """
    head = cfg.MODEL.ROI_BOX_HEAD.NAME
    if head == "FastRCNNConvFCHead":
        return FastRCNNConvFCHead(
            input_size,
            cfg.MODEL.ROI_BOX_HEAD.NUM_CONV,
            cfg.MODEL.ROI_BOX_HEAD.CONV_DIM,
            cfg.MODEL.ROI_BOX_HEAD.NUM_FC,
            cfg.MODEL.ROI_BOX_HEAD.FC_DIM,
            norm=cfg.MODEL.ROI_BOX_HEAD.NORM,
        )
    raise ValueError("Unknown head {}".format(head))

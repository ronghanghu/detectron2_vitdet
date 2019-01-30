import numpy as np
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import weight_init


class FastRCNN2MLPHead(nn.Module):
    def __init__(self, input_size, mlp_dim):
        """
        Args:
            input_size: channels, or (channels, height, width)
            mlp_dim: int
        """
        super(FastRCNN2MLPHead, self).__init__()
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)
        self.fc1 = nn.Linear(input_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)

        self._output_size = mlp_dim

        for l in [self.fc1, self.fc2]:
            weight_init.c2_xavier_fill(l)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    @property
    def output_size(self):
        return self._output_size


def build_box_head(cfg, input_size):
    """
    Args:
        input_size: channels, or (channels, height, width)
    """
    head = cfg.MODEL.ROI_BOX_HEAD.NAME
    if head == "FastRCNN2MLPHead":
        return FastRCNN2MLPHead(input_size, cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM)
    raise ValueError("Unknown head {}".format(head))

import torch.nn as nn


def c2_xavier_fill(module):
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize
    """
    # Caffe2 implementation uses XavierFill, which in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    nn.init.constant_(module.bias, 0)

import torch
from torch import nn


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called "weight" and "bias".
    The two buffers are computed from the original four parameters of BN:
    mean, variance, scale (gamma), offset (beta).
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - mean) / std * scale + offset`, but will be slightly cheaper.
    The pre-trained backbone models from Caffe2 are already in such a frozen format.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))

    def forward(self, x):
        scale = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        return x * scale + bias

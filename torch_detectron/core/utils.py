"""
Miscellaneous utility functions
"""

import torch


def nonzero(tensor):
    """
    Equivalent to numpy.nonzero(). The difference torch.nonzero()
    is that it returns a tuple of 1d tensors, and not a 2d tensor.
    This is more convenient in a few cases.
    This should maybe be sent to core pytorch
    """
    result = tensor.nonzero()
    if result.numel() > 0:
        return torch.unbind(result, 1)
    return (result,) * tensor.dim()


# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1./9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()



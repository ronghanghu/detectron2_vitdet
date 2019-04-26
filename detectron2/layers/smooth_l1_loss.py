import torch


def smooth_l1_loss(input, target, beta):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:

            | 0.5 * x ** 2 / beta   if abs(x) < beta
     f(x) = |
            | abs(x) - 0.5 * beta   otherwise,

    where x = input - target.

    Smooth L1 loss is related to Huber loss, which is defined as:

            | 0.5 * x ** 2                  if abs(x) < beta
     h(x) = |
            | beta * (abs(x) - 0.5 * beta)  otherwise

    Smooth L1 loss differs by a factor of 1 / beta in the quadratic segment
    and a factor of beta in the linear segment. This leads to the following
    differences:

     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.

    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.

    Args:
        input (Tensor): input tensor
        target (Tensor): target value tensor
        beta (float): L1 to L2 change point. For beta values < 1e-5, L1 loss is computed.

    Returns:
        The pointwise loss summed over all elements.

    Shapes:
        input: (N, *) where * means, any number of additional dimensions
        target: (N, *), same shape as the input
        output: scalar

    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
     """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss.sum()

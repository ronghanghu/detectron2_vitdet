import torch


def smooth_l1_loss(input, target, alpha_in, alpha_out, beta=1./9):
    n = torch.abs((input - target) * alpha_in)
    cond = n < beta
    loss = alpha_out * torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss.sum()



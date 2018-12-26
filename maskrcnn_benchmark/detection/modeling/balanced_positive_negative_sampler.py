import torch


def sample_with_positive_fraction(labels, num_samples, positive_fraction):
    """
    Args:
        labels (Tensor): 1D labels with -1 (ignore), 0 (negative) or positive labels .
        num_samples (int):
        positive_fraction (float):

    Returns:
        pos_idx, neg_idx (Tensor): 1D indices. The total number of indices is `num_samples`.
            The fraction of positive indices is `positive_fraction` if possible.
    """
    positive = torch.nonzero(labels >= 1).squeeze(1)
    negative = torch.nonzero(labels == 0).squeeze(1)

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx

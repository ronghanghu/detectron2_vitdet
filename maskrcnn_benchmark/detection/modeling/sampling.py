import torch


def subsample_labels(labels, num_samples, positive_fraction):
    """
    Return `num_samples` random samples from `labels`, with a fraction of
    positives no larger than `positive_fraction`.

    Args:
        labels (Tensor): (N, ) label vector with values:
              -1: ignore
               0: background ("negative") class
             > 0: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.

    Returns:
        pos_idx, neg_idx (Tensor): 1D indices. The total number of indices is `num_samples`
            if possible. The fraction of positive indices is `positive_fraction` if possible.
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

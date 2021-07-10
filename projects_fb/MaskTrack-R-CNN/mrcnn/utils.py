# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch.nn import functional as F

__all__ = ["accuracy", "weighted_cross_entropy"]


def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def weighted_cross_entropy(pred, label, weight, avg_factor=None, reduce=True):
    """
    Used in :class:`TrackHead.loss` to compute matching loss given matching
    score, labels, and weights. Pred has shape "# of proposals in current frame"
    x "# of ground-truth boxes in reference frame". This function is required
    instead of directly reducing with F.cross_entropy because weighting
    happens over the # of proposals in current frame, not over
    the # of ground-truth boxes (which are treated as classes).
    """

    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.0)

    num_labels = pred.shape[1]
    if torch.sum(label >= num_labels) > 0:
        print(f"detect mismatch: num_labels {num_labels}")
        print(f"max_label {torch.max(label)}")
        return 0.0

    raw = F.cross_entropy(pred, label, reduction="none")
    if reduce:
        return torch.sum(raw * weight) / avg_factor
    else:
        return raw * weight / avg_factor

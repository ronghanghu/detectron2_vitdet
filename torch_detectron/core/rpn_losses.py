"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from .utils import nonzero, smooth_l1_loss
from .proposal_matcher import Matcher
from .target_preparator import TargetPreparator


class RPNTargetPreparator(TargetPreparator):
    """
    This class returns labels and regression targets for the RPN
    """
    def prepare_labels(self, matched_targets_per_image, anchors_per_image):
        """
        Arguments:
            matched_targets_per_image (BBox): a BBox with the 'matched_idx' field set,
                containing the ground-truth targets aligned to the anchors,
                i.e., it contains the same number of elements as the number of anchors,
                and contains de best-matching ground-truth target element. This is
                returned by match_targets_to_anchors
            anchors_per_image (BBox object)
        """
        matched_idxs = matched_targets_per_image.get_field('matched_idxs')
        labels_per_image = matched_idxs >= 0
        labels_per_image = labels_per_image.to(dtype=torch.float32)
        # discard anchors that go out of the boundaries of the image
        labels_per_image[~anchors_per_image.get_field('visibility')] = -1

        # discard indices that are between thresholds
        inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        labels_per_image[inds_to_discard] = -1
        return labels_per_image


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """
    def __init__(self, target_preparator, fg_bg_sampler):
        """
        Arguments:
            target_preparator (an instance of TargetPreparator)
            fb_bg_sampler (an instance of BalancedPositiveNegativeSampler)
        """
        self.target_preparator = target_preparator
        self.fg_bg_sampler = fg_bg_sampler

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list of BBox)
            objectness (list of tensor)
            box_regression (list of tensor)
            targets (list of BBox)
        """
        assert len(anchors) == 1, 'only single feature map supported'
        assert len(objectness) == 1, 'only single feature map supported'
        anchors = anchors[0]
        objectness = objectness[0]
        box_regression = box_regression[0]

        labels, regression_targets = self.target_preparator(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = nonzero(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = nonzero(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        N, A, H, W = objectness.shape
        objectness = objectness.permute(0, 2, 3, 1).reshape(-1)
        box_regression = box_regression.view(N, -1, 4, H, W).permute(0, 3, 4, 1, 2)
        box_regression = box_regression.reshape(-1, 4)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                beta=1.0 / 9,
                size_average=False) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
                objectness[sampled_inds], labels[sampled_inds])

        return objectness_loss, box_loss

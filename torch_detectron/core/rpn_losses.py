"""
This file contains specific functions for computing losses on the RPN.
Many parts of this file are generic outside of RPN, and should be moved elsewhere
"""

import torch


BELOW_UNMATCHED_THRESHOLD = -1
BETWEEN_THRESHOLDS = -2

class Matcher(object):
    """
    This class assigns ground-truth elements to proposals.
    This is done via a match_quality_matrix, which contains
    the IoU between the M ground-truths and the N proposals, in a
    MxN matrix.
    It retuns a tensor of size N, containing the index of the
    ground-truth m that matches with proposal n.
    If there is no match or if the match doesn't satisfy the
    matching conditions, a negative value is returned.
    """
    def __init__(self, matched_threshold, unmatched_threshold, force_match_for_each_row=False):
        self.matched_threshold = matched_threshold
        self.unmatched_threshold = unmatched_threshold
        self.force_match_for_each_row = force_match_for_each_row

    def __call__(self, match_quality_matrix):
        matched_vals, matches = match_quality_matrix.max(0)
        below_unmatched_threshold = matched_vals < self.unmatched_threshold
        between_thresholds = ((matched_vals >= self.unmatched_threshold)
                & (matched_vals < self.matched_threshold))

        # Ross implementation always uses the box with the max overlap
        matched_backup = matches.clone()
        # TODO this is the convention of TF
        matches[below_unmatched_threshold] = BELOW_UNMATCHED_THRESHOLD
        matches[between_thresholds] = BETWEEN_THRESHOLDS

        if self.force_match_for_each_row:
            force_matched_vals, force_matches = match_quality_matrix.max(1)
            force_max_overlaps = torch.nonzero(match_quality_matrix == force_matched_vals[:, None])
            # matches[force_max_overlaps[:, 1]] = force_max_overlaps[:, 0]
            # Ross implementation always uses the box with the max overlap
            matches[force_max_overlaps[:, 1]] = matched_backup[force_max_overlaps[:, 1]]
        return matches

from core.box_ops import boxes_iou
class RPNTargetPreparator(object):
    """
    This class returns labels and regression targets for the RPN
    """
    def __init__(self, proposal_matcher, box_coder):
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder

    def match_targets_to_anchors(self, anchors, targets):
        """
        anchors: list of BBox, one for each image
        targets: list of BBox, one for each image
        """
        results = []
        for anchor, target in zip(anchors, targets):
            # TODO pass the BBox object to boxes_iou?
            match_quality_matrix = boxes_iou(target.bbox, anchor.bbox)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            matched_targets = target[matched_idxs.clamp(min=0)]
            matched_targets.add_field('matched_idxs', matched_idxs)
            results.append(matched_targets)
        return results

    def __call__(self, anchors, targets):
        # TODO assert / resize anchors to have the same .size as targets?
        matched_targets = self.match_targets_to_anchors(anchors, targets)
        labels = []
        regression_targets = []
        for matched_targets_per_image, anchors_per_image in zip(matched_targets, anchors):
            labels_per_image = matched_targets_per_image.get_field('matched_idxs') >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            labels_per_image[~anchors_per_image.get_field('visibility')] = -1
            # TODO the -2 is magic here. make this more generic
            labels_per_image[matched_targets_per_image.get_field('matched_idxs') == -2] = -1
            regression_targets_per_image = self.box_coder.encode(
                    matched_targets_per_image.bbox, anchors_per_image.bbox)
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """
    def __init__(self, batch_size_per_image, positive_fraction, which_negatives=None):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.which_negatives = which_negatives  # TODO remove?

    def __call__(self, matched_idxs):
        """
        matched idxs: list of tensors containing -1, 0 or 1 values.
        Each tensor corresponds to a specific image.
        -1 values are ignored, 0 are considered as negatives and 1 as
        positives.

        returns a the indices of the sampled positive values and negative
        values, concatenated over all images
        """
        pos_idx = []
        neg_idx = []
        offset = 0
        for matched_idxs_per_image in matched_idxs:
            positive = (matched_idxs_per_image == 1).nonzero()
            negative = (matched_idxs_per_image == 0).nonzero()

            positive = positive.squeeze(1) if positive.numel() > 0 else positive
            negative = negative.squeeze(1) if negative.numel() > 0 else negative

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)

            perm1 = torch.randperm(positive.numel())[:num_pos]
            perm2 = torch.randperm(negative.numel())[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # TODO maybe convert to byte mask representation
            # instead of adding offset?
            pos_idx.append(pos_idx_per_image + offset)
            neg_idx.append(neg_idx_per_image + offset)

            offset += matched_idxs_per_image.numel()

        pos_idx = torch.cat(pos_idx, dim=0)
        neg_idx = torch.cat(neg_idx, dim=0)

        return pos_idx, neg_idx


# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1./9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs((input - target))
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

from torch.nn import functional as F
class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """
    def __init__(self, target_preparator, fg_bg_sampler):
        self.target_preparator = target_preparator
        self.fg_bg_sampler = fg_bg_sampler

    def __call__(self, anchors, objectness, box_regression, targets):
        assert len(anchors) == 1, 'only single feature map supported'
        assert len(objectness) == 1, 'only single feature map supported'
        anchors = anchors[0]
        objectness = objectness[0]
        box_regression = box_regression[0]

        labels, regression_targets = self.target_preparator(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
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
                size_average=False) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
                objectness[sampled_inds], labels[sampled_inds])

        return objectness_loss, box_loss

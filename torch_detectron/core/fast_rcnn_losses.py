import torch
from torch.nn import functional as F

from .proposal_matcher import Matcher
from .utils import nonzero, smooth_l1_loss
from .target_preparator import TargetPreparator


class FastRCNNTargetPreparator(TargetPreparator):
    """
    This class returns labels and regression targets for Fast R-CNN
    """
    def prepare_labels(self, matched_targets_per_image, anchors_per_image):
        matched_idxs = matched_targets_per_image.get_field('matched_idxs')
        labels_per_image = matched_targets_per_image.get_field('labels')
        labels_per_image = labels_per_image.to(dtype=torch.int64)

        # discard indices that are between thresholds
        inds_to_discard = matched_idxs == Matcher.BELOW_UNMATCHED_THRESHOLD
        labels_per_image[inds_to_discard] = -1

        neg_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        labels_per_image[neg_inds] = 0
        return labels_per_image


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    """
    def __init__(self, target_preparator, fg_bg_sampler):
        """
        Arguments:
            target_preparator: an instance of TargetPreparator
            fg_bg_sampler: an instance of BalancedPositiveNegativeSampler
        """
        self.target_preparator = target_preparator
        self.fg_bg_sampler = fg_bg_sampler

    def subsample(self, anchors, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled anchors.
        Note: this function keeps a state.

        Arguments:
            anchors (list of BBox)
            targets (list of BBox)
        """
        assert len(anchors) == 1, 'only single feature map supported'
        anchors = anchors[0]

        labels, regression_targets = self.target_preparator(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = pos_inds_img | neg_inds_img
            anchors[img_idx] = anchors[img_idx][img_sampled_inds]
            sampled_inds.append(img_sampled_inds)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        self._labels = labels
        self._regression_targets = regression_targets
        self._sampled_pos_inds = sampled_pos_inds
        self._sampled_neg_inds = sampled_neg_inds
        self._sampled_inds = sampled_inds

        return [anchors]

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list of tensor)
            box_regression (list of tensor)
        """
        assert len(class_logits) == 1, 'only single feature map supported'
        class_logits = class_logits[0]
        box_regression = box_regression[0]
        device = class_logits.device

        if not hasattr(self, '_labels'):
            raise RuntimeError('subsample needs to be called before')
        labels = self._labels
        regression_targets = self._regression_targets
        sampled_pos_inds = torch.cat(self._sampled_pos_inds, dim=0)
        sampled_neg_inds = torch.cat(self._sampled_neg_inds, dim=0)
        sampled_inds = torch.cat(self._sampled_inds, dim=0)

        # delete cached elements
        for attr in ['_labels', '_regression_targets',
                '_sampled_pos_inds', '_sampled_neg_inds', '_sampled_inds']:
            delattr(self, attr)

        # get indices of the positive examples in the subsampled space
        markers = torch.arange(sampled_inds.sum(), device=device)
        marked_sampled_inds = torch.zeros(sampled_inds.shape[0],
                dtype=torch.int64, device=device)
        marked_sampled_inds[sampled_inds] = markers
        sampled_pos_inds_subset = marked_sampled_inds[sampled_pos_inds]

        sampled_pos_inds = nonzero(sampled_pos_inds)[0]
        sampled_neg_inds = nonzero(sampled_neg_inds)[0]
        sampled_inds = nonzero(sampled_inds)[0]

        classification_loss = F.cross_entropy(
                class_logits, labels[sampled_inds])

        # FIXME workaround because can't unsqueeze empty tensor in PyTorch
        # when there are no positive labels
        if len(sampled_pos_inds) == 0:
            box_loss = torch.tensor(0., device=device, requires_grad=True)
            return classification_loss, box_loss

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        labels_pos = labels[sampled_pos_inds]
        map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds_subset[:, None], map_inds],
                regression_targets[sampled_pos_inds],
                size_average=False,
                beta=1) / (sampled_inds.numel())

        return classification_loss, box_loss


# FIXME merge this with FastRCNNLossComputation
class FastRCNNOHEMLossComputation(object):
    """
    This class computes the Fast R-CNN loss
    In an OHEM manner.
    """
    def __init__(self, target_preparator, fg_bg_sampler):
        self.target_preparator = target_preparator
        self.fg_bg_sampler = fg_bg_sampler

    def __call__(self, anchors, class_logits, box_regression, targets):
        assert len(anchors) == 1, 'only single feature map supported'
        assert len(class_logits) == 1, 'only single feature map supported'
        anchors = anchors[0]
        class_logits = class_logits[0]
        box_regression = box_regression[0]

        device = class_logits.device

        labels, regression_targets = self.target_preparator(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = nonzero(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = nonzero(torch.cat(sampled_neg_inds, dim=0))[0]
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)


        classification_loss = F.cross_entropy(
                class_logits[sampled_inds], labels[sampled_inds])

        # FIXME workaround because can't unsqueeze empty tensor in PyTorch
        # when there are no positive labels
        if len(sampled_pos_inds) == 0:
            box_loss = torch.tensor(0., device=device, requires_grad=True)
            return classification_loss, box_loss

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        labels_pos = labels[sampled_pos_inds]
        map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds[:, None], map_inds],
                regression_targets[sampled_pos_inds],
                size_average=False,
                beta=1) / (sampled_inds.numel())

        return classification_loss, box_loss

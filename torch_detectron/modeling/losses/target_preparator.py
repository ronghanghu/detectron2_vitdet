import torch

from ..box_ops import boxes_iou


class TargetPreparator(object):
    """
    Base class for aligning the ground-truth targets to the anchors.
    Given a BoxList anchor and a BoxList target, this class performs the correspondences
    between the anchors and the targets, and returns the labels and the regression
    targets for all the anchors.

    Extensions should inherit from this class and define the prepare_labels
    method, which defines how to compute the labels for each anchor.
    """

    def __init__(self, proposal_matcher, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder

    def match_targets_to_anchors(self, anchors, targets):
        """
        Arguments:
            anchors: list[BoxList], one for each image
            targets: list[BoxList], one for each image
        """
        results = []
        for anchor, target in zip(anchors, targets):
            if len(anchor) > 0:
                match_quality_matrix = boxes_iou(target, anchor)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
            else:
                matched_idxs = torch.empty(
                    0, dtype=torch.int64, device=target.bbox.device
                )
            # get the targets corresponding GT for each anchor
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_targets = self.index_target(target, matched_idxs.clamp(min=0))
            matched_targets.add_field("matched_idxs", matched_idxs)
            results.append(matched_targets)
        return results

    def index_target(self, target, index):
        """
        This function is used to index the targets, possibly only propagating a few
        fields of target (instead of them all)

        Arguments:
            target (BoxList): an arbitrary bbox object, containing many possible fields
            index (tensor): the indices to select.
        """
        return target[index]

    def prepare_labels(self, matched_targets_per_image, anchors_per_image):
        """
        Arguments:
            matched_targets_per_image (BoxList): a BoxList with the 'matched_idx' field set,
                containing the ground-truth targets aligned to the anchors,
                i.e., it contains the same number of elements as the number of anchors,
                and contains de best-matching ground-truth target element. This is
                returned by match_targets_to_anchors
            anchors_per_image (a BoxList object)

        This method should return a single tensor, containing the labels
        for each element in the anchors
        """
        raise NotImplementedError("Needs to be implemented")

    def __call__(self, anchors, targets):
        """
        Arguments:
            anchors: list[BoxList], one for each image
            targets: list[BoxList], one for each image
        """
        # TODO assert / resize anchors to have the same .size as targets?
        matched_targets = self.match_targets_to_anchors(anchors, targets)
        labels = []
        regression_targets = []
        for matched_targets_per_image, anchors_per_image in zip(
            matched_targets, anchors
        ):
            labels_per_image = self.prepare_labels(
                matched_targets_per_image, anchors_per_image
            )

            regression_targets_per_image = self.box_coder.encode(
                matched_targets_per_image.bbox, anchors_per_image.bbox
            )
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

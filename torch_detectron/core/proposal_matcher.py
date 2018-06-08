import torch


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

    BELOW_UNMATCHED_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, matched_threshold, unmatched_threshold, force_match_for_each_row=False):
        """
        Arguments:
            matched_threshold (float): similarity greater or equal to this value are considered
                matched
            unmatched_threshold (float): similarity smaller than unmatched_threshold is
                unsidered unmatched.
            force_match_for_each_row (bool): if True, forces that every ground-truth element
                has at least one match, even if the match is below matched_threshold
        """
        self.matched_threshold = matched_threshold
        self.unmatched_threshold = unmatched_threshold
        self.force_match_for_each_row = force_match_for_each_row

    def __call__(self, match_quality_matrix):
        """
        Arguments:
            match_quality_matrix: a MxN tensor, containing the pairwise box similarity between
            the M ground-truth boxes and the N anchor boxes.
        """
        matched_vals, matches = match_quality_matrix.max(0)
        below_unmatched_threshold = matched_vals < self.unmatched_threshold
        between_thresholds = ((matched_vals >= self.unmatched_threshold)
                & (matched_vals < self.matched_threshold))

        # Ross implementation always uses the box with the max overlap
        matched_backup = matches.clone()
        # TODO this is the convention of TF
        matches[below_unmatched_threshold] = Matcher.BELOW_UNMATCHED_THRESHOLD
        matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS

        if self.force_match_for_each_row:
            force_matched_vals, force_matches = match_quality_matrix.max(1)
            force_max_overlaps = torch.nonzero(match_quality_matrix == force_matched_vals[:, None])
            # matches[force_max_overlaps[:, 1]] = force_max_overlaps[:, 0]
            # Ross implementation always uses the box with the max overlap
            matches[force_max_overlaps[:, 1]] = matched_backup[force_max_overlaps[:, 1]]
        return matches

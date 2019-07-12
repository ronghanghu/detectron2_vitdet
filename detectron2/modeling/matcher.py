import torch


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.

    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.

    The matcher returns a vector of length N containing the index of the ground-truth
    element m in [0, M) that matches to prediction n in [0, N). If a prediction was
    not matched to any ground-truth element, then a negative value is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): to match with any ground-truth element, a prediction
                must have a maximum match quality, over all ground-truth elements,
                that is >= this threshold.
            low_threshold (float): a lower maximum quality threshold that is used to
                stratify predictions into three levels:
                1) maximum match quality >= high_threshold
                    These predictions will be matched to a ground-truth element with
                    which they have maximum overlap.
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                    These "gray zone" predictions are not matched, but they are given
                    a special value in case downstream code wishes to treat them specially
                    (e.g., assign these predictions an ignore label).
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
                    These predictions are not matched.
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.
        """
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).

        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M) or a negative value indicating that
                prediction i could not be matched. A negative value of
                Matcher.BELOW_LOW_THREHSOLD indicates that the highest quality ground-truth
                was lower than the low_threshold. A negative value of Matcher.BETWEEN_THRESHOLDS
                indicates that the highest quality match was between low_threshold and
                high_threshold.
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            return match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.BELOW_LOW_THRESHOLD, dtype=torch.int64
            )
        assert torch.all(match_quality_matrix >= 0)

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of the
        Faster R-CNN paper: https://arxiv.org/pdf/1506.01497v3.pdf.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties.
        # Note that the matches qualities must be positive due to the use of
        # `torch.nonzero`.
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]

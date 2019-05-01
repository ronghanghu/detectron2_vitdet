import torch

from detectron2.layers import interpolate

# from PIL import Image  # TODO investigate how to avoid opencv dependency


class Keypoints:
    """
    Stores the keypoints for all objects in one image. Instances have a `keypoints` property that
        contains the x, y and visibility of each keypoint as a tensor of shape  (N, M, 3) where
        N is the number of instances, and M is the number of keypoints per instance.

    The visiblity of each keypoint may be one of three integers:
        0 - not visible
        1 - visible
        2 - occluded
    """

    def __init__(self, keypoints):
        """
        Arguments:
            keypoints: A Tensor, numpy array, or list of the x, y, and visibility of each keypoint.
                The shape should be (N, M, 3) where N is the number of
                instances, and M is the number of keypoints per instance.
        """
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device("cpu")
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        assert keypoints.dim() == 3 and keypoints.shape[2] == 3, keypoints.shape
        self.tensor = keypoints

    def __len__(self):
        return self.tensor.size(0)

    def to(self, *args, **kwargs):
        return type(self)(self.tensor.to(*args, **kwargs))

    def to_heatmap(self, boxes, heatmap_size):
        """
        Args:
            boxes: Nx4 tensor, the boxes to draw the keypoints to

        Returns:
            heatmaps: A tensor of shape (N, M) containing an integer spatial label
                in the range [0, heatmap_size**2 - 1] for each keypoint in the input.
            valid: A tensor of shape (N, M) containing whether each keypoint is in
                the roi or not.
        """
        return _keypoints_to_heatmap(self.tensor, boxes, heatmap_size)

    def __getitem__(self, item):
        """
        Create a new `Keypoints` by indexing on this `Keypoints`.

        The following usage are allowed:
        1. `new_kpts = kpts[3]`: return a `Keypoints` which contains only one instance.
        2. `new_kpts = kpts[2:10]`: return a slice of key points.
        3. `new_kpts = kpts[vector]`, where vector is a torch.ByteTensor
           with `length = len(kpts)`. Nonzero elements in the vector will be selected.

        Note that the returned Keypoints might share storage with this Keypoints,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Keypoints([self.tensor[item]])
        return Keypoints(self.tensor[item])

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s


# TODO make this nicer, this is a direct translation from C2 (but removing the inner loop)
def _keypoints_to_heatmap(keypoints, rois, heatmap_size):
    """
    Encode keypoint locations into a target heatmap for use in SoftmaxWithLoss across space.

    Maps keypoints from the half-open interval [x1, x2) on continuous image coordinates to the
    closed interval [0, heatmap_size - 1] on discrete image coordinates. We use the
    continuous-discrete conversion from Heckbert 1990 ("What is the coordinate of a pixel?"):
    d = floor(c) and c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.

    Arguments:
        keypoints: tensor of keypoint locations in of shape (N, M, 3).
        rois: Nx4 tensor of rois in xyxy format
        heatmap_size: integer side length of square heatmap.

    Returns:
        heatmaps: A tensor of shape (N, M) containing an integer spatial label
            in the range [0, heatmap_size**2 - 1] for each keypoint in the input.
        valid: A tensor of shape (N, M) containing whether each keypoint is in
            the roi or not.
    """

    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid


@torch.no_grad()
def heatmaps_to_keypoints(maps, rois):
    """
    Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, #keypoints, 4) with the last dimension corresponding to (x, y, logit, prob)
    for each keypoint.

    Converts a discrete image coordinate in an NxN image to a continuous keypoint coordinate. We
    maintain consistency with keypoints_to_heatmap by using the conversion from Heckbert 1990:
    c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.
    """
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = (rois[:, 2] - rois[:, 0]).clamp(min=1)
    heights = (rois[:, 3] - rois[:, 1]).clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    num_rois, num_keypoints = maps.shape[:2]
    xy_preds = maps.new_zeros(rois.shape[0], num_keypoints, 4)

    width_corrections = widths / widths_ceil
    height_corrections = heights / heights_ceil

    keypoints_idx = torch.arange(num_keypoints, device=maps.device)

    for i in range(num_rois):
        outsize = (int(heights_ceil[i]), int(widths_ceil[i]))
        roi_map = interpolate(maps[[i]], size=outsize, mode="bicubic", align_corners=False).squeeze(
            0
        )

        roi_map_probs = scores_to_probs(roi_map)

        w = roi_map.shape[2]
        pos = roi_map.view(num_keypoints, -1).argmax(1)

        x_int = pos % w
        y_int = (pos - x_int) // w

        assert (
            roi_map_probs[keypoints_idx, y_int, x_int]
            == roi_map_probs.view(num_keypoints, -1).max(1)[0]
        ).all()

        x = (x_int.float() + 0.5) * width_corrections[i]
        y = (y_int.float() + 0.5) * height_corrections[i]

        xy_preds[i, :, 0] = x + offset_x[i]
        xy_preds[i, :, 1] = y + offset_y[i]
        xy_preds[i, :, 2] = roi_map[keypoints_idx, y_int, x_int]
        xy_preds[i, :, 3] = roi_map_probs[keypoints_idx, y_int, x_int]

    return xy_preds


def scores_to_probs(scores):
    """ Converts a CxHxW tensor of scores to probabilities spatially."""
    num_keypoints = scores.shape[0]
    max_score, _ = scores.view(num_keypoints, -1).max(1)
    tmp = (scores - max_score.view(num_keypoints, 1, 1)).exp_()
    scores = tmp / tmp.sum((1, 2), keepdim=True)
    return scores

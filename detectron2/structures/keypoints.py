import numpy as np
import cv2
import torch

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
            a N x heatmap_size x heatmap_size heatmap.
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

    Returns: A tensor containing an integer spatial label in the range [0, heatmap_size**2 - 1] for
        each keypoint in the input. Shape is (N, M)
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


# TODO: direct copy from caffe2 detectron.
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

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = np.maximum(widths, 1)
    heights = np.maximum(heights, 1)
    widths_ceil = np.ceil(widths)
    heights_ceil = np.ceil(heights)

    # NCHW to NHWC for use with OpenCV
    maps = np.transpose(maps, [0, 2, 3, 1])
    min_size = 0  # cfg.KRCNN.INFERENCE_MIN_SIZE
    num_keypoints = 17
    xy_preds = np.zeros((len(rois), 4, num_keypoints), dtype=np.float32)
    for i in range(len(rois)):
        if min_size > 0:
            roi_map_width = int(np.maximum(widths_ceil[i], min_size))
            roi_map_height = int(np.maximum(heights_ceil[i], min_size))
        else:
            roi_map_width = widths_ceil[i]
            roi_map_height = heights_ceil[i]
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height

        # Resample the roi map to its size in the original image
        """
        PIL version, much slower
        roi_map = np.empty((int(roi_map_height), int(roi_map_width), num_keypoints), dtype=np.float32)
        for j in range(maps[i].shape[2]):
            map_image = Image.fromarray(maps[i, ..., j])
            map_image = map_image.resize((roi_map_width, roi_map_height), resample=Image.BICUBIC)
            roi_map[..., j] = np.array(map_image, copy=False)
        """
        roi_map = cv2.resize(
            maps[i], (roi_map_width, roi_map_height), interpolation=cv2.INTER_CUBIC
        )

        # Bring back to CHW
        roi_map = np.transpose(roi_map, [2, 0, 1])
        roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        for k in range(num_keypoints):
            pos = roi_map[k, :, :].argmax()
            x_int = pos % w
            y_int = (pos - x_int) // w
            assert roi_map_probs[k, y_int, x_int] == roi_map_probs[k, :, :].max()
            x = (x_int + 0.5) * width_correction
            y = (y_int + 0.5) * height_correction
            xy_preds[i, 0, k] = x + offset_x[i]
            xy_preds[i, 1, k] = y + offset_y[i]
            xy_preds[i, 2, k] = roi_map[k, y_int, x_int]
            xy_preds[i, 3, k] = roi_map_probs[k, y_int, x_int]

    return np.transpose(xy_preds, [0, 2, 1])


def scores_to_probs(scores):
    """ Converts a CxHxW tensor of scores to probabilities spatially."""
    channels = scores.shape[0]
    for c in range(channels):
        temp = scores[c, :, :]
        max_score = temp.max()
        temp = np.exp(temp - max_score) / np.sum(np.exp(temp - max_score))
        scores[c, :, :] = temp
    return scores

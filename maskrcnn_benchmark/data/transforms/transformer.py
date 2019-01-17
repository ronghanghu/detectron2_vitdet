# -*- coding: utf-8 -*-
# File: transformers.py

from PIL import Image

from .base import ImageTransformer
from .transform import ResizeTransform


__all__ = ["Flip", "Resize", "ResizeShortestEdge", "Normalize"]


class Flip(ImageTransformer):
    """
    Random flip the image either horizontally or vertically.
    """

    def __init__(self, horiz=False, vert=False, prob=0.5):
        """
        Args:
            horiz (bool): use horizontal flip.
            vert (bool): use vertical flip.
            prob (float): probability of flip.
        """
        super(Flip, self).__init__()
        if horiz and vert:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horiz and not vert:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def _get_transform_params(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        return (do, h, w)

    def _transform_image(self, img, param):
        do, _, _ = param
        if do:
            if self.horiz:
                ret = img[:, ::-1]
            else:
                ret = img[::-1, :]
        else:
            ret = img
        return ret

    def _transform_coords(self, coords, param):
        do, h, w = param
        if do:
            if self.vert:
                coords[:, 1] = h - coords[:, 1]
            else:
                coords[:, 0] = w - coords[:, 0]
        return coords


class Resize(ImageTransformer):
    """ Resize image to a target size"""

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def _get_transform_params(self, img):
        return ResizeTransform(
            img.shape[0], img.shape[1], self.shape[0], self.shape[1], self.interp
        )

    def _transform_image(self, img, t):
        return t.apply_image(img)

    def _transform_coords(self, coords, t):
        return t.apply_coords(coords)


class ResizeShortestEdge(ImageTransformer):
    """
    Try resizing the shortest edge to a certain number
    while avoiding the longest edge to exceed max_size.
    """

    def __init__(self, short_edge_length, max_size, sample_style, interp=Image.BILINEAR):
        """
        Args:
            short_edge_length ([int, int]): a [min, max] interval from which to sample the
                shortest edge length.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): select short side from a range or choose from given values
        """
        super(ResizeShortestEdge, self).__init__()

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def _get_transform_params(self, img):
        h, w = img.shape[:2]

        if self.is_range:
            size = self.rng.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = self.rng.choice(self.short_edge_length)

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)

    def _transform_image(self, img, t):
        return t.apply_image(img)

    def _transform_coords(self, coords, t):
        return t.apply_coords(coords)


class Normalize(ImageTransformer):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self._init(locals())

    def _transform_image(self, img, _):
        img = (img - self.mean).astype("float32") / self.std
        return img

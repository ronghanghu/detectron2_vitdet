# -*- coding: utf-8 -*-
# File: transform.py

from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from PIL import Image

from .base import ImageAugmentor

__all__ = []


class TransformAugmentorBase(ImageAugmentor):
    """
    Base class of augmentors which use :class:`ImageTransform`
    for the actual implementation of the transformations.

    It assumes that :meth:`_get_augment_params` should
    return a :class:`ImageTransform` instance, and it will use
    this instance to augment both image and coordinates.
    """

    def _augment(self, img, t):
        return t.apply_image(img)

    def _augment_coords(self, coords, t):
        return t.apply_coords(coords)


class ImageTransform(metaclass=ABCMeta):
    """
    A deterministic image transformation, used to implement
    the (probably random) augmentors.

    This way the deterministic part
    (the actual transformation which may be common between augmentors)
    can be separated from the random part
    (the random policy which is different between augmentors).
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def apply_image(self, img):
        pass

    @abstractmethod
    def apply_coords(self, coords):
        pass


class ResizeTransform(ImageTransform):
    def __init__(self, h, w, newh, neww, interp):
        super(ResizeTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        assert img.shape[:2] == (self.h, self.w)
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize((self.neww, self.newh), self.interp)
        ret = np.asarray(pil_image)
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.neww * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.newh * 1.0 / self.h)
        return coords


class CropTransform(ImageTransform):
    def __init__(self, h0, w0, h, w):
        super(CropTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        return img[self.h0 : self.h0 + self.h, self.w0 : self.w0 + self.w]

    def apply_coords(self, coords):
        coords[:, 0] -= self.w0
        coords[:, 1] -= self.h0
        return coords

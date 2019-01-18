# -*- coding: utf-8 -*-
# File: transform.py

import numpy as np
from abc import ABCMeta, abstractmethod
from PIL import Image


__all__ = []


class ImageTransform(metaclass=ABCMeta):
    """
    Base class for image transformation ops.
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
        """
        Args:
           will crop by img[h0:h0+h, w0:w0+w]
        """
        super(CropTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        return img[self.h0 : self.h0 + self.h, self.w0 : self.w0 + self.w]

    def apply_coords(self, coords):
        coords[:, 0] -= self.w0
        coords[:, 1] -= self.h0
        return coords

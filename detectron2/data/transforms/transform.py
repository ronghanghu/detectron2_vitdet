# -*- coding: utf-8 -*-
# File: transform.py

import numpy as np
from abc import ABCMeta, abstractmethod
from PIL import Image

__all__ = [
    "Transform",
    "TransformList",
    "ResizeTransform",
    "CropTransform",
    "NoOpTransform",
    "HFlipTransform",
]


class Transform(metaclass=ABCMeta):
    """
    Base class for implementations of __deterministic__
    transformations for image and other data structures.

    "Deterministic" requires that the output of all methods
    of this class are deterministic w.r.t their input arguments.
    In training, there should be a higher-level policy
    that generates (likely with random variations) these transform ops.

    Each transform op may handle several data types, e.g.:
    image, coordinates, segmentation, bounding boxes.
    Some of them have a default implementation,
    but can be overwritten if the default isn't appropriate.

    The implementation of each method may choose to modify its input data
    in-place for efficient transformation.
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def apply_image(self, img):
        """
        Apply the transform on an image.

        Args:
            img (ndarray): of shape HxWxC or HxW.
        """
        pass

    @abstractmethod
    def apply_coords(self, coords):
        """
        Apply the transform on coordinates.

        Args:
            coords (ndarray): of shape Nx2. Each row is (x, y)
        """
        pass

    def apply_segmentation(self, segmentation):
        """
        Apply the transform on a full-image segmentation.
        By default will just perform "apply_image".

        Args:
            segmentation (ndarray): of shape HxW.
        """
        return self.apply_image(segmentation)

    def apply_box(self, box):
        """
        Apply the transform on an axis-aligned box.
        By default will transform the corner points and use their
        minimum/maximum to create a new axis-aligned box.
        Note that this default may change the size of your box, e.g. in rotations.

        Args:
            box (ndarray): Nx4 floating point array of XYXY format
                in absolute coordinates.
        """
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1])
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)
        return trans_boxes


class TransformList:
    """
    Maintain a list of transform operations which will be applied in sequence.

    Attributes:
        transforms (list[Transform])
    """

    def __init__(self, transforms):
        """
        Args:
            transforms (list[Transform])
        """
        super().__init__()
        for t in transforms:
            assert isinstance(t, Transform), t
        self.transforms = transforms

    def _apply(self, x, meth):
        for t in self.transforms:
            x = getattr(t, meth)(x)
        return x

    def apply_image(self, x):
        return self._apply(x, "apply_image")

    def apply_coords(self, x):
        return self._apply(x, "apply_coords")

    def apply_segmentation(self, x):
        return self._apply(x, "apply_segmentation")

    def apply_box(self, x):
        return self._apply(x, "apply_box")


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        self._init(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        pil_image = Image.fromarray(img)
        interp_method = interp if interp is not None else self.interp
        pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
        ret = np.asarray(pil_image)
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class CropTransform(Transform):
    def __init__(self, x0, y0, w, h):
        """
        Args:
           will crop by img[y0:y0+h, x0:x0+w]
        """
        super().__init__()
        self._init(locals())

    def apply_image(self, img):
        return img[self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w]

    def apply_coords(self, coords):
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords


class NoOpTransform(Transform):
    """
    A transform that does nothing.
    """

    def __init__(self):
        super().__init__()

    def apply_image(self, img):
        return img

    def apply_coords(self, coords):
        return coords


class HFlipTransform(Transform):
    """
    Horizontal flip.
    """

    def __init__(self, width):
        super().__init__()
        self._init(locals())

    def apply_image(self, img):
        return img[:, ::-1]

    def apply_coords(self, coords):
        coords[:, 0] = self.width - coords[:, 0]
        return coords

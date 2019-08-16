# -*- coding: utf-8 -*-
# File: transformer.py
import inspect
import numpy as np
import pprint
import sys
from abc import ABCMeta, abstractmethod
from PIL import Image

from .transform import (
    CropTransform,
    ExtentTransform,
    HFlipTransform,
    NoOpTransform,
    ResizeTransform,
    Transform,
    TransformList,
)

__all__ = [
    "RandomCrop",
    "RandomExtent",
    "RandomFlip",
    "Resize",
    "ResizeShortestEdge",
    "TransformGen",
    "apply_transform_gens",
]


def check_dtype(img):
    assert isinstance(img, np.ndarray), "[TransformGen] Needs an numpy array, but got a {}!".format(
        type(img)
    )
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[TransformGen] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    assert img.ndim in [2, 3], img.ndim


class TransformGen(metaclass=ABCMeta):
    """
    TransformGen takes an image of type uint8 in range [0, 255], or
    floating point in range [0, 1] or [0, 255] as input.

    It creates a :class:`Transform` based on the given image, sometimes with randomness.
    The transform can then be used to transform images
    or other data (boxes, points, annotations, etc.) associated with it.

    A list of `TransformGen` can be applied with :func:`TransformGen.generate_transforms`.
    The assumption made in this class
    is that the image itself is sufficient to instantiate a transform.
    When this assumption is not true, you need to create the transforms by your own.
    """

    def __init__(self):
        self.reset()

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def reset(self):
        """ reset rng and other state """
        # TODO check if this has duplicate state between processes.
        # self.rng = get_rng(self)
        self.rng = np.random.RandomState()

    @abstractmethod
    def get_transform(self, img):
        pass

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return self.rng.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyTransformGen(field1={self.field1}, field2={self.field2})"
        """
        try:
            argspec = inspect.getargspec(self.__init__)
            assert argspec.varargs is None, "The default __repr__ doesn't work for varargs!"
            assert argspec.keywords is None, "The default __repr__ doesn't work for kwargs!"
            fields = argspec.args[1:]
            index_field_has_default = len(fields) - (
                0 if argspec.defaults is None else len(argspec.defaults)
            )

            classname = type(self).__name__
            argstr = []
            for idx, f in enumerate(fields):
                assert hasattr(self, f), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(f)
                )
                attr = getattr(self, f)
                if idx >= index_field_has_default:
                    if attr is argspec.defaults[idx - index_field_has_default]:
                        continue
                argstr.append("{}={}".format(f, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__


class RandomFlip(TransformGen):
    """
    Flip the image horizontally with the given probability.

    TODO Vertical flip to be implemented.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): probability of flip.
        """
        horiz, vert = True, False
        # TODO implement vertical flip when we need it
        super().__init__()

        if horiz and vert:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horiz and not vert:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, img):
        _, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            return HFlipTransform(w)
        else:
            return NoOpTransform()


class Resize(TransformGen):
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

    def get_transform(self, img):
        return ResizeTransform(
            img.shape[0], img.shape[1], self.shape[0], self.shape[1], self.interp
        )


class ResizeShortestEdge(TransformGen):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def get_transform(self, img):
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


class RandomCrop(TransformGen):
    """
    Randomly crop a subimage out of an image.
    """

    def __init__(self, crop_type: str, crop_size):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute".
                See `config/defaults.py` for explanation.
            crop_size (tuple[float]): the relative ratio or absolute pixels of
                height and width
        """
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute"]
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = self.rng.randint(h - croph + 1)
        w0 = self.rng.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return self.crop_size
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


class RandomExtent(TransformGen):
    """
    Outputs an image by cropping a random "subrect" of the source image.

    The subrect can be parameterized to include pixels outside the source image,
    in which case they will be set to zeros (i.e. black). The size of the output
    image will vary with the size of the random subrect.
    """

    def __init__(
        self,
        scale_range,
        shift_range,
    ):
        """
        Args:
            output_size (h, w): Dimensions of output image
            scale_range (l, h): Range of input-to-output size scaling factor
            shift_range (x, y): Range of shifts of the cropped subrect. The rect
                is shifted by [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)],
                where (w, h) is the (width, height) of the input image. Set each
                component to zero to crop at the image's center.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        img_h, img_w = img.shape[:2]

        # Initialize src_rect to fit the input image.
        src_rect = np.array([
            -0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h
        ])

        # Apply a random scaling to the src_rect.
        src_rect *= self.rng.uniform(self.scale_range[0], self.scale_range[1])

        # Apply a random shift to the coordinates origin.
        src_rect[0::2] += self.shift_range[0] * img_w * (self.rng.rand() - 0.5)
        src_rect[1::2] += self.shift_range[1] * img_h * (self.rng.rand() - 0.5)

        # Map src_rect coordinates into image coordinates (center at corner).
        src_rect[0::2] += 0.5 * img_w
        src_rect[1::2] += 0.5 * img_h

        return ExtentTransform(
            src_rect=(
                src_rect[0],
                src_rect[1],
                src_rect[2],
                src_rect[3],
            ),
            output_size=(
                int(src_rect[3] - src_rect[1]),
                int(src_rect[2] - src_rect[0]),
            ),
        )


def apply_transform_gens(transform_gens, img):
    """
    Apply a list of :class:`TransformGen` on the input image, and
    returns the transformed image and a list of transforms.

    We cannot simply create and return all transforms without
    applying it to the image, because a subsequent transform may
    need the output of the previous one.

    Args:
        transform_gens (list): list of :class:`TransformGen` instance to
            be applied.
        img (ndarray): uint8 or floating point images with 1 or 3 channels.

    Returns:
        ndarray: the transformed image
        TransformList: contain the transforms that's used.
    """
    for g in transform_gens:
        assert isinstance(g, TransformGen), g

    check_dtype(img)

    tfms = []
    for g in transform_gens:
        tfm = g.get_transform(img)
        assert isinstance(
            tfm, Transform
        ), "TransformGen {} must return an instance of Transform! Got {} instead".format(g, tfm)
        img = tfm.apply_image(img)
        tfms.append(tfm)
    return img, TransformList(tfms)

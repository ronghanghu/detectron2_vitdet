# -*- coding: utf-8 -*-
# File: base.py


import inspect
import numpy as np
import pprint
from abc import ABCMeta, abstractmethod


__all__ = ["ImageTransformer", "ImageTransformers"]


def check_dtype(img):
    assert isinstance(img, np.ndarray), "[Transformer] Needs an numpy array, but got a {}!".format(
        type(img)
    )
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[Transformer] Got image of type {}, use uint8 or floating points instead!".format(img.dtype)


class ImageTransformer(metaclass=ABCMeta):
    """
    ImageTransformer takes an image of type uint8 in range [0, 255], or
    floating point in range [0, 1] or [0, 255] as input.
    It applies image transformation on the image, and optionally
    returns the "transformation parameters" which can be used
    to transform another image, or transform the coordinates in the image.

    The implementation may choose to modify the input image or coordinates
    in-place for efficient transformation.
    """

    def __init__(self):
        self.reset_state()

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def reset_state(self):
        """ reset rng and other state """
        # TODO check if this has duplicate state between processes.
        # self.rng = get_rng(self)
        self.rng = np.random.RandomState()

    def transform_image(self, img):
        """
        Transform the input image.

        Args:
            img: input image

        Returns:
            transformed image
        """
        img, params = self._transform_image_get_params(img)
        return img

    def transform_image_get_params(self, img):
        """
        Transform the image and return the transformation parameters.
        If the transformation is non-deterministic (random),
        the returned parameters can be used to transform another image
        with the identical transformation (using `transform_with_params`, `transform_coords`).
        This can be used for, e.g. transforming image, masks, keypoints altogether with the
        same transformation.

        Args:
            img: input image

        Returns:
            (transformed image, transformation params)
        """
        return self._transform_image_get_params(img)

    def _transform_image_get_params(self, img):
        """
        Transform the image and return both image and params

        Can be overwritten by subclasses
        """
        prms = self._get_transform_params(img)
        return (self._transform_image(img, prms), prms)

    def transform_image_with_params(self, img, param):
        """
        Transform the image with the given param.
        The transform is allowed to modify image in-place.

        Args:
            img: input image
            param: transformation params returned by :meth:`transform_return_params`

        Returns:
            transformed image
        """
        return self._transform_image(img, param)

    @abstractmethod
    def _transform_image(self, img, param):
        """
        Transform the image with the given param.
        Should be overwritten by subclasses
        """

    def _get_transform_params(self, img):
        """
        Get the transformer parameters.

        Can be overwritten by subclasses.
        """
        return None

    def transform_coords(self, coords, param):
        """
        Transform the coordinates given the param.

        By default, a transformer keeps coordinates unchanged.
        If a subclass of :class:`ImageTransformer` changes coordinates but couldn't implement this method,
        it should ``raise NotImplementedError()``.

        Args:
            coords: Nx2 floating point numpy array where each row is (x, y)
            param: transformation params returned by :meth:`transform_return_params`

        Returns:
            new coords
        """
        return self._transform_coords(coords, param)

    def _transform_coords(self, coords, param):
        """
        Should be overwritten by subclasses if the transformer changes coordinates.
        """
        return coords

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
        "MyTransformer(field1={self.field1}, field2={self.field2})"
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
            return super(ImageTransformer, self).__repr__()

    __str__ = __repr__


class ImageTransformers(ImageTransformer):
    """
    Applies a list of ImageTransformers on the input image.
    """

    def __init__(self, transformers):
        """
        Args:
            transformers (list): list of :class:`ImageTransformer` instance to
            be applied.
        """
        self.transformers = transformers
        super(ImageTransformers, self).__init__()

    def _get_transform_params(self, img):
        # the next transformer requires the previous one to finish
        raise RuntimeError(
            "Cannot simply get all parameters of a ImageTransformers without running the transformation!"
        )

    def _transform_image_get_params(self, img):
        check_dtype(img)
        assert img.ndim in [2, 3], img.ndim

        prms = []
        for a in self.transformers:
            img, prm = a._transform_image_get_params(img)
            prms.append(prm)
        return img, prms

    def _transform_image(self, img, param):
        check_dtype(img)
        assert img.ndim in [2, 3], img.ndim
        for tfm, prm in zip(self.transformers, param):
            img = tfm._transform_image(img, prm)
        return img

    def _transform_coords(self, coords, param):
        for tfm, prm in zip(self.transformers, param):
            coords = tfm._transform_coords(coords, prm)
        return coords

    def reset_state(self):
        """ Will reset state of each transformer """
        for a in self.transformers:
            a.reset_state()

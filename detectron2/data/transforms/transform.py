# -*- coding: utf-8 -*-
# File: transform.py

import inspect
import numpy as np
from abc import ABCMeta, abstractmethod
from PIL import Image

__all__ = [
    "BlendTransform",
    "CropTransform",
    "ExtentTransform",
    "HFlipTransform",
    "NoOpTransform",
    "Transform",
    "TransformList",
    "ResizeTransform",
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
            img (ndarray): of shape HxWxC or HxW. The array can be of type uint8 in
                range [0, 255], or floating point in range [0, 1] or [0, 255].
        """
        pass

    @abstractmethod
    def apply_coords(self, coords):
        """
        Apply the transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).
        """
        pass

    def apply_segmentation(self, segmentation):
        """
        Apply the transform on a full-image segmentation.
        By default will just perform "apply_image".

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer or bool dtype.
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

    def apply_polygons(self, polygons):
        """
        Apply the transform on a list of polygons,
        each represented by a Nx2 array.
        By default will just transform all the points.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point
                array of (x, y) format in absolute coordinates.
        """
        return [self.apply_coords(p) for p in polygons]

    @classmethod
    def register_type(cls, data_type: str, func):
        """
        Register the given function as
        a handler that this transform will use for a specific data type.

        Args:
            data_type (str): the name of the data type (e.g., box)
            func (callable): takes a transform and a data, returns the transformed data.

        Examples:
            def func(flip_transform, voxel_data):
                return transformed_voxel_data
            HFlipTransform.register_handler("voxel", func)

            # ...
            transform = HFlipTransform(...)
            transform.apply_voxel(voxel_data)  # func will be called
        """
        assert callable(
            func
        ), "You can only register a callable to a Transform. Got {} instead.".format(func)
        argspec = inspect.getfullargspec(func)
        assert len(argspec.args) == 2, (
            "You can only register a function that takes two positional "
            "arguments to a Transform! Got a function with spec {}".format(str(argspec))
        )
        setattr(cls, "apply_" + data_type, func)


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

    def __getattr__(self, name):
        if name.startswith("apply_"):
            return lambda x: self._apply(x, name)
        raise AttributeError("TransformList object has no attribute {}".format(name))

    def __add__(self, other):
        other = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(self.transforms + other)

    def __iadd__(self, other):
        other = other.transforms if isinstance(other, TransformList) else [other]
        self.transforms.extend(other)
        return self

    def __radd__(self, other):
        other = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(other + self.transforms)


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

    def apply_polygons(self, polygons):
        import shapely.geometry as geometry

        # Create a window that will be used to crop
        crop_box = geometry.box(self.x0, self.y0, self.x0 + self.w, self.y0 + self.h).buffer(0.0)

        cropped_polygons = []

        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0.0)
            # polygon must be valid to perform intersection.
            assert polygon.is_valid, polygon
            cropped = polygon.intersection(crop_box)
            if cropped.is_empty:
                continue
            if not isinstance(cropped, geometry.collection.BaseMultipartGeometry):
                cropped = [cropped]
            # one polygon may be cropped to multiple ones
            for poly in cropped:
                # It could produce lower dimensional objects like lines or points,
                # which we want to ignore
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                    continue
                coords = np.asarray(poly.exterior.coords)
                # NOTE This process will produce an extra identical vertex at the end.
                # So we remove it. This is tested by `tests/test_data_transform.py`
                cropped_polygons.append(coords[:-1])
        return [self.apply_coords(p) for p in cropped_polygons]


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


class ExtentTransform(Transform):
    """
    Extracts a subregion from the source image and scales it to the output size.

    The fill color is used to map pixels from the source rect that fall outside
    the source image.

    See: https://pillow.readthedocs.io/en/latest/PIL.html#PIL.ImageTransform.ExtentTransform
    """

    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
        """
        Args:
            src_rect (x0, y0, x1, y1): src coordinates
            output_size (h, w): dst image size
            interp: PIL interpolation methods
            fill: Fill color used when src_rect extends outside image
        """
        super().__init__()
        self._init(locals())

    def apply_image(self, img, interp=None):
        h, w = self.output_size
        ret = Image.fromarray(img).transform(
            size=(w, h),
            method=Image.EXTENT,
            data=self.src_rect,
            resample=interp if interp else self.interp,
            fill=self.fill,
        )
        return np.asarray(ret)

    def apply_coords(self, coords):
        # Transform image center from source coordinates into output coordinates
        # and then map the new origin to the corner of the output image.
        h, w = self.output_size
        x0, y0, x1, y1 = self.src_rect
        new_coords = coords.astype(np.float32)
        new_coords[:, 0] -= 0.5 * (x0 + x1)
        new_coords[:, 1] -= 0.5 * (y0 + y1)
        new_coords[:, 0] *= w / (x1 - x0)
        new_coords[:, 1] *= h / (y1 - y0)
        new_coords[:, 0] += 0.5 * w
        new_coords[:, 1] += 0.5 * h
        return new_coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class BlendTransform(Transform):
    """
    Transforms pixel colors with PIL enhance functions.
    """

    def __init__(self, src_image, src_weight, dst_weight):
        """
        Blends the input image (dst_image) with the src_image using formula:
            src_weight * src_image + dst_weight * dst_image
        Args:
            src_image (ndarray or float): Input image is blended with this image
            src_weight (float): Blend weighting of src_image
            dst_weight (float): Blend weighting of dst_image
        """
        super().__init__()
        self._init(locals())

    def apply_image(self, img, interp=None):
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = self.src_weight * self.src_image + self.dst_weight * img
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return self.src_weight * self.src_image + self.dst_weight * img

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

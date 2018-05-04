from PIL import Image
import random
import math

import torch
from torchvision.transforms import functional as F

from lib.bounding_box import BBox


"""
Proposals for new transforms: v2

We have a new BBox class, which provides a PIL.Image-compatible
interface for operations on bounding boxes.
Thus, many of the implementations in torchvision only require minor
modifications, and can be used almost as-is for bounding boxes.

For the functional API, we have two options:
  1 - either have a dispatch function that accepts an arbitrary number of inputs
    image, boxes = hflip(image, boxes)
  2 - call each function individually
    image = hflip(image)
    boxes = hflip(boxes)

Note that we can have the same functions dispatch to both images and bboxes
without much overhead, because BBox has a subset of the API of PIL.Image.

"""


#############################################################
# Option 1:
#############################################################

def hflip1(*args):
    new_args = tuple(i.transpose(Image.FLIP_LEFT_RIGHT) for i in args)
    if len(new_args) == 1:
        return new_args[0]
    return new_args

def _resize(image, min_size, max_size, interpolation):
    w, h = image.size
    min_original_size = float(min((w, h)))
    max_original_size = float(max((w, h)))
    new_max_size = max_original_size / min_original_size * min_size
    if new_max_size > max_size:
        min_size = int(round(max_size * min_original_size / max_original_size))
        new_max_size = max_size
    size = (min_size, new_max_size) if w < h else (new_max_size, min_size)
    return image.resize(size, interpolation)


def resize1(*args, size=None, max_size=math.inf, interpolation=Image.BILINEAR):
    if size is None:
        raise ValueError('expected size to be passed as an argument')
    return tuple(_resize(i, size, max_size, interpolation) for i in args)
    # new_args = tuple(F.resize(i, size, interpolation) for i in args)
    if len(new_args) == 1:
        return new_args[0]
    return new_args


class MyTransform1(object):
    def __init__(self, flip_prob, min_size, max_size,
            normalization_mean=None, normalization_std=None):
        self.flip_prob = flip_prob
        self.min_size = min_size
        self.max_size = max_size
        if normalization_mean is None:
            normalization_mean = [0.485, 0.456, 0.406]
        if normalization_std is None:
            normalization_std = [0.229, 0.224, 0.225]
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std

    def __call__(self, image, bbox, labels):
        if random.random() < self.flip_prob:
            image, bbox = hflip1(image, bbox)
        image, bbox = resize1(image, bbox, size=self.min_size, max_size=self.max_size)

        # TODO make to_tensor also support BBOx
        image, bbox = F.to_tensor(image), bbox.bbox
        image = F.normalize(image, mean=self.normalization_mean,
                std=self.normalization_std)

        return image, bbox, labels

"""
Downsides of option 1:
    - we need a boiler-plate function for dispatching to multiple inputs
"""

#############################################################
# Option 2: slighly more verbose
#############################################################

def hflip2(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def resize2(image, size, max_size=math.inf, interpolation=Image.BILINEAR):
    return _resize(image, size, max_size, interpolation)

class MyTransform2(object):
    def __init__(self, flip_prob, min_size, max_size,
            normalization_mean=None, normalization_std=None):
        self.flip_prob = flip_prob
        self.min_size = min_size
        self.max_size = max_size
        if normalization_mean is None:
            normalization_mean = [0.485, 0.456, 0.406]
        if normalization_std is None:
            normalization_std = [0.229, 0.224, 0.225]
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std

    def __call__(self, image, bbox, labels):
        if random.random() < self.flip_prob:
            image = hflip2(image)
            bbox = hflip2(bbox)
        image = resize2(image, size=self.min_size, max_size=self.max_size)
        bbox = resize2(bbox, size=self.min_size, max_size=self.max_size)

        # TODO make to_tensor also support BBOx
        image, bbox = F.to_tensor(image), bbox.bbox
        image = F.normalize(image, mean=self.normalization_mean,
                std=self.normalization_std)

        return image, bbox, labels

"""
Downsides of option 2:
    - more verbose and repetitive than option 1
"""

if __name__ == '__main__':
    import numpy as np
    image = (np.random.rand(100, 200, 3) * 255).astype(np.uint8)
    image = Image.fromarray(image)
    bbox = torch.Tensor([[0, 0, 50, 50], [1, 10, 70, 40]])
    bbox = BBox(bbox, image_width=200, image_height=100)

    transform1 = MyTransform1(0.5, 200, 300)
    transform2 = MyTransform2(0.5, 200, 300)
    print('old', image, bbox)
    print('new', transform1(image, bbox, None))
    print('new', transform2(image, bbox, None))

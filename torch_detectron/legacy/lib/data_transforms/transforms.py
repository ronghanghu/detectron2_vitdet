import random
from PIL import Image

import torch
from torchvision.transforms import functional as F

import lib.utils.box as box_utils


class Resize(object):
    def __init__(self, min_size, max_size, interpolation=Image.BILINEAR):
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation

    # def __call__(self, image, boxes=None, *args):
    def __call__(self, args):
        image, boxes = args[:2]
        w, h = image.size
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        min_size = self.min_size
        if max_original_size / min_original_size * min_size > self.max_size:
            min_size = int(round(self.max_size * min_original_size / max_original_size))
        image = F.resize(image, min_size, self.interpolation)
        ratio = min_size / min_original_size

        if boxes is not None:
            boxes = boxes * ratio

        return (image, boxes) + args[2:]


def _flip_boxes_v0(boxes, image_width):
    xmin, ymin, xmax, ymax = boxes.split(1, dim=1)
    flipped_xmin = image_width - xmax
    flipped_xmax = image_width - xmin
    flipped_boxes = torch.cat((flipped_xmin, ymin, flipped_xmax, ymax), dim=1)
    return flipped_boxes

def _flip_boxes(boxes, image_width):
    flipped_boxes = boxes.clone()
    flipped_boxes[:, 0::4] = image_width - boxes[:, 2::4]
    flipped_boxes[:, 2::4] = image_width - boxes[:, 0::4]
    return flipped_boxes


class RandomHorizontalFlip(object):
    # def __call__(self, image, boxes=None):
    def __call__(self, args):
        # TODO replace with torch.rand
        image, boxes = args[:2]
        flip = random.random() > 0.5
        if flip:
            image = F.hflip(image)
            # decide on the bbox format
            if boxes is not None:
                w, h = image.size
                boxes = _flip_boxes(boxes, w)
        return (image, boxes) + args[2:]


class ClipBoxes(object):
    def __call__(self, image, boxes):
        # preproceesing the boxes
        height = image.size(1)
        width= image.size(2)
        boxes = box_utils.clip_boxes_to_image(
            boxes, height, width
        )
        return image, boxes


class UniqueBoxes(object):
    def __call__(self, image, boxes):
        keep = box_utils.unique_boxes(boxes)
        boxes = boxes[keep, :]
        return image, boxes


class FilterSmallBoxes(object):
    def __init__(self, min_proposal_size):
        self.min_proposal_size = min_proposal_size

    def __call__(self, image, boxes):
        keep = box_utils.filter_small_boxes(boxes, self.min_proposal_size)
        boxes = boxes[keep, :]
        return image, boxes

# Hack!!!
class MyTransf(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, args):
        image = args[0]
        image = self.transforms(image)
        return (image,) + args[1:]


from torchvision.transforms import functional as F
import random
class MyTransform(object):
    def __init__(self, flip_prob, min_size, max_size,
            normalization_mean=None, normalization_std=None, to_bgr255=False):
        self.flip_prob = flip_prob
        self.min_size = min_size
        self.max_size = max_size
        if normalization_mean is None:
            normalization_mean = [0.485, 0.456, 0.406]
        if normalization_std is None:
            normalization_std = [0.229, 0.224, 0.225]
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        image = F.resize(image, size=self.min_size, max_size=self.max_size)
        target = F.resize(target, size=self.min_size, max_size=self.max_size)

        # TODO make to_tensor also support BBOx
        image = F.to_tensor(image)
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.normalization_mean,
                std=self.normalization_std)

        return image, target





DEFAULT_CLASSES = ('__background__', # always index 0
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor')

class VOCAnnotations(object):
    def __init__(self, keep_difficult, class_names=DEFAULT_CLASSES):
        self.keep_difficult = keep_difficult
        self.class_map = dict(zip(class_names, range(len(class_names))))

    def __call__(self, anno):
        objs = anno['annotation']['object']
        boxes, classes = zip(*[[[obj['bndbox']['xmin'], obj['bndbox']['ymin'], obj['bndbox']['xmax'], obj['bndbox']['ymax']],
            self.class_map[obj['name']]] for obj in objs 
            if obj['difficult'] == '0' or self.keep_difficult])
        boxes = torch.FloatTensor(boxes)
        classes = torch.LongTensor(classes)

        return boxes, classes

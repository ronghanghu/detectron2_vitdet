import torch
import torchvision
from torchvision.transforms import functional as F
import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        image = F.resize(image, self.min_size, max_size=self.max_size)
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean=None, std=None, to_bgr255=True):
        # TODO remove those values?
        if mean is None:
            mean = [102.9801, 115.9465, 122.7717]
        if std is None:
            std = [1, 1, 1]
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean,
                std=self.std)
        return image, target
        

class ImageTransform(object):
    """
    Data transformations to be performed in the image and the targets.
    The normalization is specific for the C2 pretrained models
    """
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, x, target):
        x = F.resize(x, 800, max_size=1333)
        target = target.resize(x.size)
        if random.random() < self.flip_prob:
            x = F.hflip(x)
            target = target.transpose(0)
        x = F.to_tensor(x)

        x = x[[2, 1, 0]] * 255
        x -= torch.tensor([102.9801, 115.9465, 122.7717]).view(3,1,1)

        return x, target


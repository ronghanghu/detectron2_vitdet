# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import division

import torch


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensor, image_sizes):
        """
        Arguments:
            tensor (Tensor): of shape (N, C, H, W)
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w).
        """
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self):
        return len(self.image_sizes)

    def to(self, *args, **kwargs):
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @staticmethod
    def from_tensors(tensors, size_divisible=0):
        """
        Args:
            tensors: a tuple or list of `torch.Tensors`, each of shape (C, Hi, Wi).
                The Tensors will be padded with zeros so that they will have the same shape.
            size_divisible (int): If `size_divisible > 0`, also adds padding to ensure
                the common height and width is divisible by `size_divisible`

        Returns:
            an `ImageList`.
        """
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.dim() == 3 and t.shape[0] == tensors[0].shape[0], t.shape
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        # TODO Ideally, just remove this and let the model handle arbitrary input sizes
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        image_sizes = [im.shape[-2:] for im in tensors]

        return ImageList(batched_imgs, image_sizes)

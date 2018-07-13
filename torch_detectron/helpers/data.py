"""
Convenience files for obtaining datasets and dataloaders.
Specific to COCO-like dataset
"""

import bisect

import torch.utils.data
import torch.utils.data.distributed
import torch.utils.data.sampler

from torch_detectron.datasets.coco import COCODataset
from torch_detectron.helpers.config_utils import ConfigNode
from torch_detectron.utils import data_transforms as T
from torch_detectron.utils.concat_dataset import ConcatDataset
from torch_detectron.utils.data_collate import BatchCollator
from torch_detectron.utils.data_samplers import GroupedBatchSampler
from torch_detectron.utils.data_samplers import compute_aspect_ratios

MEAN = [102.9801, 115.9465, 122.7717]
STD = [1., 1., 1.]


class _COCODataset(ConfigNode):
    """
    FILES: a list of 2-element tuples, containing the
        path to the annotations file and the path to the
        data directory
    """

    def __call__(self, dataset_transform):
        """
        dataset_transform: an object that will perform data
            transformations to the image and targets of the
            dataset.
        """
        dataset_files = self.FILES
        assert isinstance(dataset_files, (list, tuple))
        assert all(len(l) == 2 for l in dataset_files)

        # keep all images for testing, and remove images that do not
        # contain any positive instances during training
        remove_images_without_annotations = self.config.TRAIN.DATA.DATASET == self

        dataset_list = []
        for ann_file, data_dir in dataset_files:
            dataset = COCODataset(
                ann_file, data_dir, remove_images_without_annotations, dataset_transform
            )
            dataset_list.append(dataset)

        dataset = dataset_list[0]
        if len(dataset_list) > 1:
            dataset = ConcatDataset(dataset_list)

        return dataset


class _DataLoader(ConfigNode):
    """
    SAMPLER:
    IMAGES_PER_BATCH:
    COLLATOR:
    NUM_WORKERS:
    """

    def __call__(self, dataset):
        sampler = self.SAMPLER(dataset)
        batch_sampler = self.BATCH_SAMPLER(dataset, sampler)
        collator = self.COLLATOR()
        num_workers = self.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        return data_loader


class _DataSampler(ConfigNode):
    """
    SHUFFLE:
    DISTRIBUTED:
    """

    def __call__(self, dataset):
        shuffle = self.SHUFFLE
        distributed = self.DISTRIBUTED
        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
            if distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        return sampler


def _quantize(x, bins):
    bins = sorted(bins.copy())
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


class _BatchDataSampler(ConfigNode):
    def __call__(self, dataset, sampler):
        aspect_grouping = self.ASPECT_GROUPING
        images_per_batch = self.IMAGES_PER_BATCH
        if aspect_grouping:
            if not isinstance(aspect_grouping, (list, tuple)):
                aspect_grouping = [aspect_grouping]
            aspect_ratios = compute_aspect_ratios(dataset)
            group_ids = _quantize(aspect_ratios, aspect_grouping)
            batch_sampler = GroupedBatchSampler(
                sampler, group_ids, images_per_batch, drop_uneven=False
            )
        else:
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler, images_per_batch, drop_last=False
            )
        return batch_sampler


class _Collator(ConfigNode):
    """
    SIZE_DIVISIBLE:
    """

    def __call__(self):
        return BatchCollator(self.SIZE_DIVISIBLE)


class _DataTransform(ConfigNode):
    def __call__(self):
        min_size = self.MIN_SIZE
        max_size = self.MAX_SIZE
        mean = self.MEAN
        std = self.STD
        to_bgr255 = self.TO_BGR255
        flip_prob = self.FLIP_PROB

        resize_transform = T.Resize(min_size, max_size)
        normalize_transform = T.Normalize(mean=mean, std=std, to_bgr255=to_bgr255)

        transform = T.Compose(
            [
                resize_transform,
                T.RandomHorizontalFlip(flip_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )
        return transform


class _Data(ConfigNode):
    def __call__(self):
        transform = self.TRANSFORM()
        dataset = self.DATASET(transform)
        data_loader = self.DATALOADER(dataset)
        return data_loader

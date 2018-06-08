"""
Convenience files for obtaining datasets and dataloaders.
Specific to COCO-like dataset
"""


import torch.utils.data
from torch.utils.data.dataset import ConcatDataset
import torch.utils.data.distributed
import torch.utils.data.sampler

from torch_detectron.datasets.coco import COCODataset

from torch_detectron.utils import data_transforms as T
from torch_detectron.utils.data_collate import BatchCollator

from torch_detectron.helpers.config_utils import ConfigClass


class _COCODataset(ConfigClass):
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

        dataset_list = []
        for ann_file, data_dir in dataset_files:
            dataset = COCODataset(ann_file, data_dir, dataset_transform)
            dataset_list.append(dataset)

        dataset = dataset_list[0]
        if len(dataset_list) > 1:
            dataset = ConcatDataset(dataset_list)

        return dataset

class _DataLoader(ConfigClass):
    """
    SAMPLER:
    IMAGES_PER_BATCH:
    COLLATOR:
    NUM_WORKERS:
    """
    def __call__(self, dataset):
        sampler = self.SAMPLER(dataset)
        images_per_batch = self.IMAGES_PER_BATCH
        collator = self.COLLATOR()
        num_workers = self.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(dataset,
                    batch_size=images_per_batch, num_workers=num_workers,
                    sampler=sampler, collate_fn=collator)
        return data_loader

class _DataSampler(ConfigClass):
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

class _Collator(ConfigClass):
    """
    SIZE_DIVISIBLE:
    """
    def __call__(self):
        return BatchCollator(self.SIZE_DIVISIBLE)

class _DataTransform(ConfigClass):
    def __call__(self):
        min_size = self.MIN_SIZE
        max_size = self.MAX_SIZE
        mean = self.MEAN
        std = self.STD
        to_bgr255 = self.TO_BGR255
        flip_prob = self.FLIP_PROB

        resize_transform = T.Resize(min_size, max_size)
        normalize_transform = T.Normalize(
                mean=mean, std=std, to_bgr255=to_bgr255)

        transform = T.Compose([
            resize_transform,
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform
        ])
        return transform

class _Data(ConfigClass):
    def __call__(self):
        transform = self.TRANSFORM()
        dataset = self.DATASET(transform)
        data_loader = self.DATALOADER(dataset)
        return data_loader

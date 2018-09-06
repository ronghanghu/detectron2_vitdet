import bisect

import torch.utils.data

from torch_detectron.utils.data_collate import BatchCollator
from torch_detectron.utils.data_samplers import GroupedBatchSampler
from torch_detectron.utils.data_samplers import compute_aspect_ratios
from torch_detectron.datasets.coco import COCODataset
from torch_detectron.utils import data_transforms as T
from torch_detectron.utils.concat_dataset import ConcatDataset

from torch_detectron.config.paths_catalog import DatasetCatalog


def make_transform(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    resize_transform = T.Resize(min_size, max_size)

    to_bgr255 = True
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            resize_transform,
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform


def make_coco_dataset(cfg, is_train=True):
    key = "TRAIN" if is_train else "TEST"
    dataset_list = getattr(cfg.DATASETS, key)

    transforms = make_transform(cfg, is_train)

    datasets = []

    for dataset_name in dataset_list:
        annotation_path, folder = DatasetCatalog.get(dataset_name)
        dataset = COCODataset(
            annotation_path,
            folder,
            remove_images_without_annotations=is_train,
            transforms=transforms,
        )
        datasets.append(dataset)

    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return dataset


def make_data_sampler(dataset, shuffle, distributed):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        assert (
            distributed == False
        ), "Distributed with no shuffling on the dataset not supported"
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = sorted(bins.copy())
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_batch):
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


def make_data_loader(cfg, is_train=True):
    if is_train:
        images_per_batch = cfg.DATALOADER.IMAGES_PER_BATCH_TRAIN
        shuffle = True
    else:
        images_per_batch = cfg.DATALOADER.IMAGES_PER_BATCH_TEST
        shuffle = False

    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
    distributed = False  # FIXME

    dataset = make_coco_dataset(cfg, is_train)
    sampler = make_data_sampler(dataset, shuffle, distributed)
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, aspect_grouping, images_per_batch
    )
    collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )
    return data_loader

import bisect
import copy
import itertools
import logging
import numpy as np
import torch.utils.data
from termcolor import colored

from detectron2.data import DatasetFromList, MapDataset, MetadataCatalog, samplers
from detectron2.utils.comm import get_world_size

from tabulate import tabulate

from .dataset_catalog import DatasetCatalog
from .transforms import DetectionTransform

__all__ = ["build_detection_train_loader", "build_detection_test_loader"]


def filter_images_with_only_crowd_annotations(dataset_dicts):
    """
    Filter out images with only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if ann["iscrowd"] == 0:
                return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def filter_images_with_few_keypoints(dataset_dicts, min_keypoints_per_image):
    """
    Filter out images with too few number of keypoints.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def visible_keypoints_in_image(dic):
        # Each keypoints field has the format [x1, y1, v1, ...], where v is visibility
        annotations = dic["annotations"]
        return sum(
            (np.array(ann["keypoints"][2::3]) > 0).sum()
            for ann in annotations
            if "keypoints" in ann
        )

    dataset_dicts = [
        x for x in dataset_dicts if visible_keypoints_in_image(x) >= min_keypoints_per_image
    ]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with fewer than {} keypoints.".format(
            num_before - num_after, min_keypoints_per_image
        )
    )
    return dataset_dicts


def _quantize(x, bin_edges):
    bin_edges = copy.copy(bin_edges)
    bin_edges = sorted(bin_edges)
    quantized = list(map(lambda y: bisect.bisect_right(bin_edges, y), x))
    return quantized


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = [x["category_id"] - 1 for x in annos if not x["iscrowd"]]
        histogram += np.histogram(classes, bins=hist_bins)[0]
    data = [[class_names[i], v] for i, v in enumerate(histogram)]
    data.append(["total ({} categories)".format(num_classes), sum(x[1] for x in data)])
    table = tabulate(data, headers=["category", "#instances"], tablefmt="pipe")
    logger = logging.getLogger(__name__)
    logger.info(
        "Distribution of categories among all training instances:\n" + colored(table, "cyan")
    )


def build_batch_data_sampler(
    sampler, images_per_batch, group_bin_edges=None, grouping_features=None
):
    """
    Return a dataset index sampler that batches dataset indices possibly with
    grouping to improve training efficiency.

    Args:
        sampler (torch.utils.data.sampler.Sampler): any subclass of
            :class:`torch.utils.data.sampler.Sampler`.
        images_per_batch (int): the batch size. Note that the sampler may return
            batches that have between 1 and images_per_batch (inclusive) elements
            because the underlying index set (and grouping partitions, if grouping
            is used) may not be divisible by images_per_batch.
        group_bin_edges (None, list[number], tuple[number]): If None, then grouping
            is disabled. If a list or tuple is given, the values are used as bin
            edges for defining len(group_bin_edges) + 1 groups. When batches are
            sampled, only elements from the same group are returned together.
        grouping_features (None, list[number], tuple[number]): If None, then grouping
            is disabled. If a list or tuple is given, it must specify for each index
            in the underlying dataset the value to be used for placing that dataset
            index into one of the grouping bins.

    Returns:
        A BatchSampler or subclass of BatchSampler.
    """
    if group_bin_edges and grouping_features:
        assert isinstance(group_bin_edges, (list, tuple))
        assert isinstance(grouping_features, (list, tuple))
        group_ids = _quantize(grouping_features, group_bin_edges)
        batch_sampler = samplers.GroupedBatchSampler(sampler, group_ids, images_per_batch)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=True
        )  # drop last so the batch always have the same size
        # NOTE when we add batch inference support, make sure not to use this.
    return batch_sampler


def build_detection_train_loader(cfg, start_iter=0):
    """
    Returns:
        a torch DataLoader object
    """
    num_gpus = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
        images_per_batch % num_gpus == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
        images_per_batch, num_gpus
    )
    images_per_gpu = images_per_batch // num_gpus

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When training with more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH. You must also adjust the learning rate and "
            "schedule length according to the linear scaling rule. See for "
            "example: http://git.io/fhSc4"
        )

    assert len(cfg.DATASETS.TRAIN)
    dataset_dicts = list(
        itertools.chain.from_iterable(DatasetCatalog.get(split) for split in cfg.DATASETS.TRAIN)
    )

    dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if cfg.MODEL.KEYPOINT_ON:
        min_kp = cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if min_kp > 0:
            dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_kp)

    try:
        dataset_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).dataset_name
        class_names = MetadataCatalog.get(dataset_name).class_names
        print_instances_class_histogram(dataset_dicts, class_names)
    except AttributeError:  # class names are not available for this dataset
        pass

    dataset = DatasetFromList(dataset_dicts, copy=False)

    # Bin edges for batching images with similar aspect ratios. If ASPECT_RATIO_GROUPING
    # is enabled, we define two bins with an edge at height / width = 1.
    group_bin_edges = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
    aspect_ratios = [float(img["height"]) / float(img["width"]) for img in dataset]

    dataset = MapDataset(dataset, DetectionTransform(cfg, True))

    sampler = samplers.TrainingSampler(len(dataset), seed=start_iter)
    batch_sampler = build_batch_data_sampler(
        sampler, images_per_gpu, group_bin_edges, aspect_ratios
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def build_detection_test_loader(cfg, dataset_name):
    """
    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
            dataset, with test-time transformation and batching.
    """
    dataset = DatasetFromList(DatasetCatalog.get(dataset_name))
    dataset = MapDataset(dataset, DetectionTransform(cfg, False))

    sampler = samplers.InferenceSampler(len(dataset))
    # Always use 1 image per GPU during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

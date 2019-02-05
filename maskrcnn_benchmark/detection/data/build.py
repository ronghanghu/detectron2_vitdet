import bisect
import copy
import logging
import torch.utils.data
from torch.utils.data.dataset import ConcatDataset

from maskrcnn_benchmark.data import MapDataset, datasets as D, samplers
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationList
from maskrcnn_benchmark.utils.comm import get_world_size

from ..config import paths_catalog
from .transforms import DetectionTransform


def build_dataset(dataset_list, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError("dataset_list should be a list of strings, got {}".format(dataset_list))
    assert len(dataset_list)
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]

        if data["factory"] == "COCODetection":
            # for COCODataset, we want to remove images without annotations during training
            args["remove_images_without_annotations"] = is_train
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train or len(datasets) <= 1:
        return datasets
    else:
        # for training, concatenate all datasets into a single one
        return [ConcatDataset(datasets)]


def build_data_sampler(dataset, shuffle, distributed):
    # "samplers" have a bad __init__ interface; they take a dataset, but internally
    # only make use of the dataset length
    if distributed:
        # NB: the distributed sampler always shuffles data
        assert shuffle  # make sure there are no surprises
        return torch.utils.data.distributed.DistributedSampler(dataset)

    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bin_edges):
    bin_edges = copy.copy(bin_edges)
    bin_edges = sorted(bin_edges)
    quantized = list(map(lambda y: bisect.bisect_right(bin_edges, y), x))
    return quantized


def build_batch_data_sampler(
    sampler,
    images_per_batch,
    group_bin_edges=None,
    grouping_features=None,
    max_iters=None,
    start_iter=0,
):
    """
    Return a dataset index sampler that batches dataset indices.

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
        max_iters (None or int): If an int, then the batch sampler will produce max_iters
            count batches, possibly cycling through the underlying sampler multiple times.
            If None, then only a single epoch of the underlying sampler is used.
        start_iter (int): If max_iters is specified, this determines the iteration at
            which to start.

    Returns:
        A BatchSampler or subclass of BatchSampler.
    """
    if group_bin_edges and grouping_features:
        assert isinstance(group_bin_edges, (list, tuple))
        assert isinstance(grouping_features, (list, tuple))
        group_ids = _quantize(grouping_features, group_bin_edges)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if max_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iters, start_iter)
    return batch_sampler


def build_detection_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus
        )
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        max_iters = cfg.SOLVER.MAX_ITER
    else:
        # Always use 1 image per GPU during inference since this is the
        # standard when reporting inference time in papers.
        images_per_batch = num_gpus
        images_per_gpu = 1
        # The distributed sampler always suffles (there's no sequential option)
        shuffle = False if not is_distributed else True
        max_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When training with more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH. You must also adjust the learning rate and "
            "schedule length according to the linear scaling rule. See for "
            "example: using://git.io/fhSc4."
        )

    # Bin edges for batching images with similar aspect ratios. If ASPECT_RATIO_GROUPING
    # is enabled, we define two bins with an edge at height / width = 1.
    group_bin_edges = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    datasets = build_dataset(dataset_list, DatasetCatalog, is_train)

    data_loaders = []
    for dataset in datasets:
        aspect_ratios = [
            float(img["original_height"]) / float(img["original_width"]) for img in dataset
        ]

        dataset = MapDataset(dataset, DetectionTransform(cfg, is_train))

        sampler = build_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = build_batch_data_sampler(
            sampler, images_per_gpu, group_bin_edges, aspect_ratios, max_iters, start_iter
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=DetectionBatchCollator(is_train, cfg.DATALOADER.SIZE_DIVISIBILITY),
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders


class DetectionBatchCollator:
    """
    Batch collator will run inside the dataloader processes.
    This batch collator batch the detection images and labels into torch.Tensor.

    Note that it's best to do this step in the dataloader process, instead of in
    the main process, because Pytorch's dataloader is less efficient when handling
    large numpy arrays rather than torch.Tensor.
    """

    def __init__(self, is_train, size_divisible):
        self.size_divisible = size_divisible
        self.is_train = is_train

    def __call__(self, dataset_dicts):
        """
        Args:
            dataset_dicts (list[dict]): each dict contains keys "image" and
                "annotations", produced by :class:`DetectionTransform`.

        Returns:
            images: ImageList
            targets: list[BoxList]. None when not training.
            dataset_dicts: the rest of the dataset_dicts
        """
        # important to remove the numpy images so that they will not go through
        # the dataloader and hurt performance
        numpy_images = [x.pop("image") for x in dataset_dicts]
        images = [torch.as_tensor(x.transpose(2, 0, 1).astype("float32")) for x in numpy_images]
        images = to_image_list(images, self.size_divisible)

        if not self.is_train:
            return images, None, dataset_dicts

        targets = []
        for dataset_dict in dataset_dicts:
            image_size = (dataset_dict["transformed_width"], dataset_dict["transformed_height"])

            annos = dataset_dict.pop("annotations")
            boxes = [obj["bbox"] for obj in annos]
            boxes = torch.as_tensor(boxes).reshape(-1, 4)
            target = BoxList(boxes, image_size, mode="xywh").convert("xyxy")

            classes = [obj["category_id"] for obj in annos]
            classes = torch.tensor(classes)
            target.add_field("labels", classes)

            masks = [obj["segmentation"] for obj in annos]
            masks = SegmentationList(masks, image_size)
            target.add_field("masks", masks)
            target = target.clip_to_image(remove_empty=True)
            targets.append(target)
        return images, targets, dataset_dicts

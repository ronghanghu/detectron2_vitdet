import bisect
import copy
import logging
import torch.utils.data

from maskrcnn_benchmark.data import ConcatDataset, MapDataset, datasets as D, samplers
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationList
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file

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


def make_data_sampler(dataset, shuffle, distributed):
    # "samplers" have a bad __init__ interface ... because only the length of dataset is used by the samplers
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def make_batch_data_sampler(
    aspect_ratios, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
    return batch_sampler


def make_detection_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file("maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True)
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    datasets = build_dataset(dataset_list, DatasetCatalog, is_train)

    data_loaders = []
    for dataset in datasets:
        ratios = []
        for i in range(len(dataset)):
            img = dataset[i]
            ratios.append(float(img["height"]) / float(img["width"]))

        dataset = MapDataset(dataset, DetectionTransform(cfg, is_train))

        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            ratios, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
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

    Note that it's best to do this step in the dataloader process, instead of in the main process,
    because Pytorch's dataloader is less efficient when handling large numpy arrays rather than torch.Tensor.
    """

    def __init__(self, is_train, size_divisible):
        self.size_divisible = size_divisible
        self.is_train = is_train

    def __call__(self, roidbs):
        """
        Args:
            roidbs (list[dict]): each contains "image" and "annotations", produced by :class:`DetectionTransform`.

        Returns:
            images: ImageList
            targets: list[BoxList]. None when not training.
            roidbs: the rest of the roidbs
        """
        # important to remove the numpy images so that they will not go through the dataloader and hurt performance
        numpy_images = [x.pop("image") for x in roidbs]
        images = [torch.as_tensor(x.transpose(2, 0, 1).astype("float32")) for x in numpy_images]
        images = to_image_list(images, self.size_divisible)

        if not self.is_train:
            return images, None, roidbs

        targets = []
        for image, roidb in zip(numpy_images, roidbs):
            image_size = image.shape[1], image.shape[0]

            annos = roidb.pop("annotations")
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
        return images, targets, roidbs

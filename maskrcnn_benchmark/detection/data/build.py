import bisect
import copy
import logging
import torch.utils.data
from torch.utils.data.dataset import ConcatDataset

from maskrcnn_benchmark.data import MapDataset, datasets as D, samplers
from maskrcnn_benchmark.structures.bounding_box import Boxes, Instances
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.keypoints import Keypoints
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationList
from maskrcnn_benchmark.utils.comm import get_world_size

from ..config import paths_catalog
from .transforms import DetectionTransform


def build_dataset(dataset_name, is_train=True, min_keypoints_per_image=0):
    """
    Args:
        dataset_name (str): The name of the dataset, handled by the DatasetCatalog,
            e.g., coco_2014_train.
        is_train (bool): whether to setup the dataset for training or testing

    Returns:
        a COCODetection instance (to be made more general).
    """
    data = paths_catalog.DatasetCatalog.get(dataset_name)
    factory = getattr(D, data["factory"])  # TODO get rid of this
    args = data["args"]

    if data["factory"] == "COCODetection":
        # for COCODataset, we want to remove images without annotations during training
        args["remove_images_without_annotations"] = is_train
        args["min_keypoints_per_image"] = min_keypoints_per_image

    # make dataset from factory
    dataset = factory(**args)
    return dataset


def _quantize(x, bin_edges):
    bin_edges = copy.copy(bin_edges)
    bin_edges = sorted(bin_edges)
    quantized = list(map(lambda y: bisect.bisect_right(bin_edges, y), x))
    return quantized


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
            sampler, images_per_batch, drop_last=False
        )
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
            "example: using://git.io/fhSc4."
        )

    min_kp = 0
    if cfg.MODEL.KEYPOINT_ON:
        min_kp = cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE

    datasets = [build_dataset(dataset_name, True, min_kp) for dataset_name in cfg.DATASETS.TRAIN]
    dataset = ConcatDataset(datasets)

    # Bin edges for batching images with similar aspect ratios. If ASPECT_RATIO_GROUPING
    # is enabled, we define two bins with an edge at height / width = 1.
    group_bin_edges = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
    aspect_ratios = [
        float(img["original_height"]) / float(img["original_width"]) for img in dataset
    ]

    dataset = MapDataset(dataset, DetectionTransform(cfg, True))

    sampler = samplers.TrainingSampler(len(dataset), seed=start_iter)
    batch_sampler = build_batch_data_sampler(
        sampler, images_per_gpu, group_bin_edges, aspect_ratios
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=DetectionBatchCollator(True, cfg.DATALOADER.SIZE_DIVISIBILITY),
    )
    return data_loader


def build_detection_test_loader(cfg, dataset):
    """
    Args:
        cfg: the yacs config
        dataset: a torch Dataset

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
            dataset, with test-time transformation and batching.
    """
    dataset = MapDataset(dataset, DetectionTransform(cfg, False))

    sampler = samplers.InferenceSampler(len(dataset))
    # Always use 1 image per GPU during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=DetectionBatchCollator(False, cfg.DATALOADER.SIZE_DIVISIBILITY),
    )
    return data_loader


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
            instances: list[Instances]. None when not training.
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
            image_size = (dataset_dict["transformed_height"], dataset_dict["transformed_width"])

            annos = dataset_dict.pop("annotations")
            boxes = [obj["bbox"] for obj in annos]
            boxes = torch.as_tensor(boxes).reshape(-1, 4)
            target = Instances(image_size)
            boxes = target.gt_boxes = Boxes(boxes, mode="xywh").clone(mode="xyxy")
            boxes.clip(image_size)

            classes = [obj["category_id"] for obj in annos]
            classes = torch.tensor(classes)
            target.gt_classes = classes

            masks = [obj["segmentation"] for obj in annos]
            masks = SegmentationList(
                masks, image_size[::-1]
            )  # TODO(yuxinwu) seg still takes (w, h)
            target.gt_masks = masks

            kpts = [obj.get("keypoints", []) for obj in annos]
            kpts = Keypoints(kpts, image_size[::-1])  # TODO(yuxinwu) kpt still takes (w, h)
            target.gt_keypoints = kpts

            target = target[boxes.nonempty()]
            targets.append(target)
        return images, targets, dataset_dicts

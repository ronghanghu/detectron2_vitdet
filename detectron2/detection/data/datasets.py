"""Centralized catalog of paths."""

import copy
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, load_sem_seg

__all__ = ["register_coco_instances", "register_coco_panoptic"]


def register_coco_instances(key, dataset_name, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        key (str): the key that identifies a split of a dataset, e.g. "coco_2014_train".
        dataset_name (str): the name of the dataset, e.g. "coco"
        json_file (str): path to the json instance annotation file
        image_root (str): directory which contains all the images
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(key, lambda: load_coco_json(json_file, image_root, dataset_name))

    # 2. Optionally, add metadata about this split,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(key).set(
        dataset_name=dataset_name, json_file=json_file, image_root=image_root, evaluator_type="coco"
    )


def register_coco_panoptic(key, dataset_name, image_root, json_file, seg_root):
    """
    Register a COCO panoptic segmentation dataset, as well as a semantic
    segmentation dataset named `key + '_stuffonly'`.

    Note:
        The added panoptic segmentation dataset uses polygons from the COCO
        instances annotations, rather than the masks from the COCO panoptic annotations.
        We now uses the polygons because it is what the PanopticFPN paper uses.
        We may later need to add an option to use masks.

        The two format have small differences:
        Polygons in the instance annotations may have overlaps.
        The mask annotations are produced by labeling the depth ordering of overlapped polygons.

    Args:
        key (str): the key that identifies a split of a dataset,
            e.g. "coco_2017_train_panoptic".
        dataset_name (str): the name of the dataset, e.g. "coco"
        image_root (str): directory which contains all the images
        json_file (str): path to the json instance annotation file
        seg_root (str): directory which contains all the ground truth segmentation annotations.
    """
    DatasetCatalog.register(
        key,
        lambda: merge_to_panoptic(
            load_coco_json(json_file, image_root, dataset_name), load_sem_seg(seg_root, image_root)
        ),
    )
    MetadataCatalog.get(key).set(
        dataset_name=dataset_name,
        json_file=json_file,
        seg_root=seg_root,
        image_root=image_root,
        evaluator_type="panoptic_seg",
    )

    semantic_key = key + "_stuffonly"
    DatasetCatalog.register(semantic_key, lambda: load_sem_seg(seg_root, image_root))
    MetadataCatalog.get(semantic_key).set(
        dataset_name=dataset_name, gt_root=seg_root, image_root=image_root, evaluator_type="sem_seg"
    )


def merge_to_panoptic(detection_dicts, semantic_segmentation_dicts):
    """
    Create dataset dicts for panoptic segmentation, by
    merging two dicts using "file_name" field to match their entries.

    Args:
        detection_dicts (list[dict]): lists of dicts for object detection or instance segmentation.
        semantic_segmentation_dicts (list[dict]): lists of dicts for semantic segmentation.

    Returns:
        list[dict] (one per input image): Each dict contains all (key, value) pairs from dicts in
            both detection_dicts and semantic_segmentation_dicts that correspond to the same image.
            The function assumes that the same key in different dicts has the same value.
    """
    results = []
    semseg_file_to_entry = {x["file_name"]: x for x in semantic_segmentation_dicts}

    for det_dict in detection_dicts:
        dic = copy.copy(det_dict)
        dic.update(semseg_file_to_entry[dic["file_name"]])
        results.append(dic)
    return results


# ======================= Predefined datasets and splits ======================


def _add_predefined_metadata():
    # coco:
    meta = MetadataCatalog.get("coco")
    # fmt: off
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    # TODO rename to instance_json_id_to_contiguous_id or thing_...
    meta.json_id_to_contiguous_id = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}  # noqa
    # 80 names for COCO instance categories:
    # TODO rename to instance_class_names or thing_class_names
    meta.class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]  # noqa

    # Mapping from contiguous stuff id (in [0, 53], used in models)
    # to id in the dataset (used for processing results)
    meta.stuff_contiguous_id_to_dataset_id = {0: 255, 1: 92, 2: 93, 3: 95, 4: 100, 5: 107, 6: 109, 7: 112, 8: 118, 9: 119, 10: 122, 11: 125, 12: 128, 13: 130, 14: 133, 15: 138, 16: 141, 17: 144, 18: 145, 19: 147, 20: 148, 21: 149, 22: 151, 23: 154, 24: 155, 25: 156, 26: 159, 27: 161, 28: 166, 29: 168, 30: 171, 31: 175, 32: 176, 33: 177, 34: 178, 35: 180, 36: 181, 37: 184, 38: 185, 39: 186, 40: 187, 41: 188, 42: 189, 43: 190, 44: 191, 45: 192, 46: 193, 47: 194, 48: 195, 49: 196, 50: 197, 51: 198, 52: 199, 53: 200}  # noqa
    # 54 names for COCO panoptic stuff categories
    meta.stuff_class_names = ["things", "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", "water-other", "window-blind", "window-other", "tree-merged", "fence-merged", "ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged", "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged", "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged", "rug-merged"]  # noqa
    # fmt: on

    # coco_person:
    meta = MetadataCatalog.get("coco_person")
    meta.class_names = ["person"]
    # TODO add COCO keypoint names, or are they needed at all?

    # cityscapes:
    meta = MetadataCatalog.get("cityscapes")
    # We choose this order because it is consistent with our old json annotation files
    # TODO Perhaps switch to an order that's consistent with Cityscapes'
    # original label, when we don't need the legacy jsons any more.
    meta.class_names = ["bicycle", "motorcycle", "rider", "train", "car", "person", "truck", "bus"]


# We hard-coded some metadata for common datasets. This will enable:
# 1. Consistency check when loading the datasets
# 2. Use models on these standard datasets directly without having the dataset annotations
_add_predefined_metadata()


# Some predefined datasets in COCO format
_PREDEFINED_SPLITS = {}
_PREDEFINED_SPLITS["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
}

_PREDEFINED_SPLITS["cityscapes"] = {
    # TODO understand what is "filtered"
    "cityscapes_fine_instanceonly_seg_train_cocostyle": (
        "cityscapes/images",
        "cityscapes/annotations/instancesonly_gtFine_train.json",
    ),
    "cityscapes_fine_instanceonly_seg_val_cocostyle": (
        "cityscapes/images",
        "cityscapes/annotations/instancesonly_filtered_gtFine_val.json",
    ),
    "cityscapes_fine_instanceonly_seg_test_cocostyle": (
        "cityscapes/images",
        "cityscapes/annotations/instancesonly_gtFine_test.json",
    ),
}

_PREDEFINED_SPLITS["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014_100.json",
    ),
}

for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS.items():
    for key, (image_root, json_file) in splits_per_dataset.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            dataset_name,
            os.path.join("datasets", json_file),
            os.path.join("datasets", image_root),
        )

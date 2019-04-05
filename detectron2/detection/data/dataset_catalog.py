"""Centralized catalog of paths."""

import os

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import load_coco_json, load_sem_seg

__all__ = ["DatasetCatalog"]


class DatasetCatalog(object):
    """
    A catalog that stores information about the splits of datasets and how to obtain them.

    It contains a mapping from strings
    (which are the names of a dataset split, e.g. "coco_2014_train")
    to a function which parses the dataset and returns the samples in the
    format of `list[dict]` in Detectron2 Dataset format (See DATASETS.md for details).

    The purpose of having this catalog is to make it easy to choose
    different datasets, by just using the strings in the config.
    """

    _REGISTERED_SPLITS = {}

    @staticmethod
    def register(key, func):
        """
        Args:
            key (str): the key that identifies a split of a dataset, e.g. "coco_2014_train".
            func (callable): a callable which takes no arguments and returns a list of dicts.
        """
        DatasetCatalog._REGISTERED_SPLITS[key] = func

    @staticmethod
    def get(key):
        """
        Call the registered function and return its results.

        Args:
            key (str): the key that identifies a split of a dataset, e.g. "coco_2014_train".

        Returns:
            list[dict]: dataset annotations in Detectron2 format.
        """
        return DatasetCatalog._REGISTERED_SPLITS[key]()

    @staticmethod
    def register_coco_format(key, dataset_name, json_file, image_root):
        """
        Register a dataset in COCO's json annotation format.

        This is an example of how to register a new dataset.
        For a dataset of a different format,
        you need to call `register()` to register it,
        and you may want to add some useful metadata to `MetadataCatalog` as well.

        Args:
            key (str): the key that identifies a split of a dataset, e.g. "coco_2014_train".
            dataset_name (str): the name of the dataset, e.g. "coco"
            json_file (str): path to the json annotation file
            image_root (str): directory which contains all the images
        """
        # 1. register a function which returns dicts
        DatasetCatalog.register(key, lambda: load_coco_json(json_file, image_root, dataset_name))

        # 2. add metadata about this split, since they will be useful in evaluation or visualization
        MetadataCatalog.get(key).set(
            dataset_name=dataset_name, json_file=json_file, image_root=image_root
        )

    @staticmethod
    def register_sem_seg_format(key, dataset_name, gt_root, image_root):
        """
        Register a dataset in semantic segmentation format.

        Args:
            key (str): the key that identifies a split of a dataset, e.g. "coco_stuff_2017_train".
            dataset_name (str): the name of the dataset, e.g. "coco"
            gt_root (str): directory which contains all the ground truth images
            image_root (str): directory which contains all the images
        """
        DatasetCatalog.register(key, lambda: load_sem_seg(gt_root, image_root))
        MetadataCatalog.get(key).set(
            dataset_name=dataset_name, gt_root=gt_root, image_root=image_root
        )


# ======================= Predefined datasets and splits ======================


def _add_predefined_metadata():
    # coco:
    meta = MetadataCatalog.get("coco")
    # fmt: off
    # Mapping from the incontiguous COCO category id to an id in [1, 80]
    meta.json_id_to_contiguous_id = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}  # noqa
    # 80 names for COCO
    meta.class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]  # noqa
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

    # coco panoptic stuff:
    # fmt: off
    meta = MetadataCatalog.get("coco_panoptic_stuff")
    meta.train_id_to_dataset_id = {0: 255, 1: 92, 2: 93, 3: 95, 4: 100, 5: 107, 6: 109, 7: 112, 8: 118, 9: 119, 10: 122, 11: 125, 12: 128, 13: 130, 14: 133, 15: 138, 16: 141, 17: 144, 18: 145, 19: 147, 20: 148, 21: 149, 22: 151, 23: 154, 24: 155, 25: 156, 26: 159, 27: 161, 28: 166, 29: 168, 30: 171, 31: 175, 32: 176, 33: 177, 34: 178, 35: 180, 36: 181, 37: 184, 38: 185, 39: 186, 40: 187, 41: 188, 42: 189, 43: 190, 44: 191, 45: 192, 46: 193, 47: 194, 48: 195, 49: 196, 50: 197, 51: 198, 52: 199, 53: 200}  # noqa
    # 54 names for COCO panoptic stuff categories
    meta.class_names = [ "things", "banner", "blanket", "bridge", "cardboard", "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light", "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood", "water-other", "window-blind", "window-other", "tree-merged", "fence-merged", "ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged", "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged", "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged", "rug-merged"]  # noqa
    # fmt: on


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
        DatasetCatalog.register_coco_format(
            key,
            dataset_name,
            os.path.join("datasets", json_file),
            os.path.join("datasets", image_root),
        )

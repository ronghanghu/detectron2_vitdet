"""Centralized catalog of paths."""

import os

from detectron2.data.datasets import load_coco_json


__all__ = ["DatasetCatalog"]


class DatasetCatalog(object):
    """
    A catalog that maps dataset names to dicts in COCO format.
    """

    _REGISTERED_DATASETS = {}
    _REGISTERED_COCO_FORMAT_DATASETS = {}

    @staticmethod
    def register(name, func):
        """
        Args:
            name (str): a name of the dataset, e.g. "coco_2014_train".
            func (callable): a callable which takes no arguments and returns a list of dicts in COCO's format.
        """
        DatasetCatalog._REGISTERED_DATASETS[name] = func

    @staticmethod
    def register_coco_format(name, json_file, image_root):
        """
        Register a dataset in COCO's json annotation format.
        Args:
            name (str): a name of the dataset, e.g. "coco_2014_train".
            json_file (str): path to the json annotation file
            image_root (str): directory which contains all the images
        """
        DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root))
        DatasetCatalog._REGISTERED_COCO_FORMAT_DATASETS[name] = (image_root, json_file)

    @staticmethod
    def get_coco_path(name):
        """
        Returns path information for COCO's json format.
        Only useful for COCO's json format.
        Should not be called if `name` was not registered by `register_coco_format`.

        Returns:
            dict: with keys "json_file" and "image_root".
        """
        image_root, json_file = DatasetCatalog._REGISTERED_COCO_FORMAT_DATASETS[name]
        return {"json_file": json_file, "image_root": image_root}

    @staticmethod
    def get(name):
        """
        Args:
            name (str): The name of the dataset, handled by the DatasetCatalog, e.g., coco_2014_train.

        Returns:
            list[dict]: dataset annotations in COCO format
        """
        return DatasetCatalog._REGISTERED_DATASETS[name]()


_PREDEFINED_DATASETS = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    # TODO understand what is "filtered"
    "cityscapes_fine_instanceonly_seg_train_cocostyle": (
        "cityscapes/images",
        "cityscapes/annotations/instancesonly_filtered_gtFine_train.json",
    ),
    "cityscapes_fine_instanceonly_seg_val_cocostyle": (
        "cityscapes/images",
        "cityscapes/annotations/instancesonly_filtered_gtFine_val.json",
    ),
    "cityscapes_fine_instanceonly_seg_test_cocostyle": (
        "cityscapes/images",
        "cityscapes/annotations/instancesonly_filtered_gtFine_test.json",
    ),
    # TODO understand what is "mod"
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "annotations/person_keypoints_train2017_train_mod.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "annotations/person_keypoints_val2017_mod.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "annotations/person_keypoints_train2017_valminusminival_mod.json",
    ),
}


# Assume pre-defined datasets live in `./datasets`.
for name, (image_root, json_file) in _PREDEFINED_DATASETS.items():
    DatasetCatalog.register_coco_format(
        name, os.path.join("datasets", json_file), os.path.join("datasets", image_root)
    )

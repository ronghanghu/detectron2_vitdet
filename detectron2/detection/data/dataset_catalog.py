"""Centralized catalog of paths."""

import os

from detectron2.data.datasets import MetadataCatalog, load_coco_json

__all__ = ["DatasetCatalog"]


class DatasetCatalog(object):
    """
    A catalog that stores information about the splits of datasets and how to obtain them.

    It contains a mapping from strings
    (which are the names of a dataset split, e.g. "coco_2014_train")
    to:

    1. A function which parses the dataset and returns the samples in the
       format of `list[dict]` in Detectron2 Dataset format (See DATASETS.md for details).
    2. The name of the dataset the returned samples belong to, e.g. "coco".

    The purpose of having this catalog is to make it easy to choose
    different datasets, by just using the strings in the config.
    """

    _REGISTERED_SPLITS = {}
    _REGISTERED_COCO_FORMAT_SPLITS = {}
    _REGISTERED_METADATA = {}

    @staticmethod
    def register(key, dataset_name, func):
        """
        Args:
            key (str): the key that identifies a split of a dataset, e.g. "coco_2014_train".
            dataset_name (str): the name of the dataset, e.g., "coco".
            func (callable): a callable which takes no arguments and returns a list of dicts.
        """
        DatasetCatalog._REGISTERED_SPLITS[key] = func
        DatasetCatalog._REGISTERED_METADATA[key] = MetadataCatalog.get(dataset_name)

    @staticmethod
    def register_coco_format(key, dataset_name, json_file, image_root):
        """
        Register a dataset in COCO's json annotation format.

        Args:
            key (str): the key that identifies a split of a dataset, e.g. "coco_2014_train".
            dataset_name (str): the name of the dataset, e.g. "coco"
            json_file (str): path to the json annotation file
            image_root (str): directory which contains all the images
        """
        DatasetCatalog.register(
            key, dataset_name, lambda: load_coco_json(json_file, image_root, dataset_name)
        )
        DatasetCatalog._REGISTERED_COCO_FORMAT_SPLITS[key] = (image_root, json_file)

    @staticmethod
    def get_coco_path(key):
        """
        Returns path information for COCO's json format.
        Only useful for json annotations in COCO's annotation format.
        Should not be called if `key` was not registered by `register_coco_format`.

        Args:
            key (str): the key that identifies a split of a dataset, e.g. "coco_2014_train".

        Returns:
            dict: with keys "json_file" and "image_root".
        """
        image_root, json_file = DatasetCatalog._REGISTERED_COCO_FORMAT_SPLITS[key]
        return {"json_file": json_file, "image_root": image_root}

    @staticmethod
    def get(key):
        """
        Args:
            key (str): the key that identifies a split of a dataset, e.g. "coco_2014_train".

        Returns:
            list[dict]: dataset annotations in Detectron2 format.
        """
        return DatasetCatalog._REGISTERED_SPLITS[key]()

    @staticmethod
    def get_metadata(key):
        """
        Args:
            key (str): the key that identifies a split of a dataset, e.g. "coco_2014_train".

        Returns:
            Metadata: the metadata instance associated with the dataset
        """
        return DatasetCatalog._REGISTERED_METADATA[key]


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

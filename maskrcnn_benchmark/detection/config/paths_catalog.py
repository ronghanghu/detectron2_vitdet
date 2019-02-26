"""Centralized catalog of paths."""

import os

from maskrcnn_benchmark.data.datasets import load_coco_json


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


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "FAIR/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_PATH_FORMAT = (
        "{prefix}/{url}/output/train/{dataset}/generalized_rcnn/model_final.pkl"
    )  # noqa B950

    C2_DATASET_COCO = "coco_2014_train%3Acoco_2014_valminusminival"
    C2_DATASET_COCO_KEYPOINTS = "keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival"

    # format: {model_name} -> part of the url
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "35857197/12_2017_baselines/e2e_faster_rcnn_R-50-C4_1x.yaml.01_33_49.iAX0mXvW",  # noqa B950
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "35857345/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml.01_36_30.cUF7QR7I",  # noqa B950
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "35857890/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_1x.yaml.01_38_50.sNxI7sX7",  # noqa B950
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "36761737/12_2017_baselines/e2e_faster_rcnn_X-101-32x8d-FPN_1x.yaml.06_31_39.5MIHi1fZ",  # noqa B950
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "35858791/12_2017_baselines/e2e_mask_rcnn_R-50-C4_1x.yaml.01_45_57.ZgkA7hPB",  # noqa B950
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "35858933/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC",  # noqa B950
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "35861795/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml.02_31_37.KqyEK4tT",  # noqa B950
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "36761843/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_1x.yaml.06_35_59.RZotkLKI",  # noqa B950
        "48616381/e2e_mask_rcnn_R-50-FPN_2x_gn": "GN/48616381/04_2018_gn_baselines/e2e_mask_rcnn_R-50-FPN_2x_gn_0416.13_23_38.bTlTI97Q",  # noqa B950
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "37697547/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_1x.yaml.08_42_54.kdzV35ao",  # noqa B950
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog._get_c2_detectron_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog._get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def _get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/") :]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def _get_c2_detectron_baselines(name):
        name = name[len("Caffe2Detectron/COCO/") :]
        url = ModelCatalog.C2_DETECTRON_MODELS[name]
        if "keypoint_rcnn" in name:
            dataset = ModelCatalog.C2_DATASET_COCO_KEYPOINTS
        else:
            dataset = ModelCatalog.C2_DATASET_COCO

        # Detectron C2 models are stored in the structure defined in `C2_DETECTRON_PATH_FORMAT`.
        url = ModelCatalog.C2_DETECTRON_PATH_FORMAT.format(
            prefix=ModelCatalog.S3_C2_DETECTRON_URL, url=url, dataset=dataset
        )
        return url

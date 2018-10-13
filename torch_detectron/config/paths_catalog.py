"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"

    DATASETS = {
        "coco_2014_train": (
            "coco/train2014",
            "coco/annotations/instances_train2014.json",
        ),
        "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
        "coco_2014_minival": (
            "coco/val2014",
            "coco/annotations/instances_minival2014.json",
        ),
        "coco_2014_valminusminival": (
            "coco/val2014",
            "coco/annotations/instances_valminusminival2014.json",
        ),
    }

    @staticmethod
    def get(name):
        return (
            os.path.join(DatasetCatalog.DATA_DIR, DatasetCatalog.DATASETS[name][1]),
            os.path.join(DatasetCatalog.DATA_DIR, DatasetCatalog.DATASETS[name][0]),
        )


class ModelCatalog(object):
    DATA_DIR = "models"
    MODELS = {"R-50": "R-50.pth", "R-101": "R-101.pth", "X-101-32x8d": "X-101-32x8d.pth"}

    @staticmethod
    def get(name):
        return os.path.join(ModelCatalog.DATA_DIR, ModelCatalog.MODELS[name])

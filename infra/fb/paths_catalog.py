"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "/mnt/vol/gfsai-east/ai-group/datasets/json_dataset_annotations"

    DATASETS = {
        "coco_2014_train": (
            "/data/local/packages/ai-group.coco_train2014/prod/coco_train2014",
            "coco/instances_train2014.json",
        ),
        "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
        "coco_2014_minival": (
            "/data/local/packages/ai-group.coco_val2014/prod/coco_val2014",
            "coco/instances_minival2014.json",
        ),
        "coco_2014_valminusminival": (
            "/data/local/packages/ai-group.coco_val2014/prod/coco_val2014",
            "coco/instances_valminusminival2014.json",
        ),
    }

    @staticmethod
    def get(name):
        return (
            os.path.join(DatasetCatalog.DATA_DIR, DatasetCatalog.DATASETS[name][1]),
            DatasetCatalog.DATASETS[name][0]
            # os.path.join(DatasetCatalog.DATA_DIR, DatasetCatalog.DATASETS[name][0]),
        )


class ModelCatalog(object):
    DATA_DIR = "/mnt/vol/gfsai-east/ai-group/users/kaiminghe/data/imagenet_models/"
    MODELS = {"R-50": "ResNet50_weights.pkl"}

    @staticmethod
    def get(name):
        return os.path.join(ModelCatalog.DATA_DIR, ModelCatalog.MODELS[name])

"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = os.path.join(
        os.path.dirname(__file__), "../../../detectron.pytorch/configs/datasets"
    )

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
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
                ann_file=os.path.join(data_dir, attrs[1]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    DATA_DIR = "/private/home/fmassa/imagenet_detectron_models/"
    MODELS = {"R-50": "R-50.pth", "R-101": "R-101.pkl", "X-101-32x8d": "X-101-32x8d.pkl"}

    @staticmethod
    def get(name):
        return os.path.join(ModelCatalog.DATA_DIR, ModelCatalog.MODELS[name])

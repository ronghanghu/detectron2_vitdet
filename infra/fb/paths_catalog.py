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
    DATA_DIR_X_101_32x8d = "/mnt/vol/gfsai-east/ai-group/users/vision/flow_run_archive/35924270/kaiminghe/20171220/DBG_IN1K_resnet_8gpu_baseline_X101_32x8d_IN1k_90epochs.16_08_16.GeZKbFAU/checkpoints"
    MODELS = {"R-50": "ResNet50_weights.pkl", "R-101": "ResNet101_weights.pkl", "X-101-32x8d": "converted_c2_model_iter415415.pkl"}

    @staticmethod
    def get(name):
        if name == "X-101-32x8d":
            return os.path.join(ModelCatalog.DATA_DIR_X_101_32x8d, ModelCatalog.MODELS[name])
        return os.path.join(ModelCatalog.DATA_DIR, ModelCatalog.MODELS[name])

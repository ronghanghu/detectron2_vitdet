"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "/mnt/vol/gfsai-east/ai-group/datasets/json_dataset_annotations"

    DATASETS = {
        # This version of coco train2014 contains a single corrupted image:
        # COCO_train2014_000000167126.jpg
        # We have used it for training all models before and during most of 2018
        # We are keeping it around for historical purposes (in case it's ever
        # useful to have for debugging, for example)
        "coco_2014_train_corrupted": (
            "/data/local/packages/ai-group.coco_train2014/prod/coco_train2014",
            "coco/instances_train2014.json",
        ),
        # This version of coco train2014 was freshly downloaded on Oct 15, 2018
        # and fixed the single corrupted image
        "coco_2014_train": (
            "/mnt/fair/coco_train2014_oct_15_2018.img",
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

    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ResNet50_weights.pkl",
        "MSRA/R-101": "ResNet101_weights.pkl",
        "FAIR/20171220/X-101-32x8d": "converted_c2_model_iter415415.pkl",
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.DATA_DIR
        if "X-101-32x8d" in name:
            prefix = ModelCatalog.DATA_DIR_X_101_32x8d
        name = name[len("ImageNetPretrained/") :]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = os.path.join(prefix, name)
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        raise RuntimeError("12_2017 Detectron Baselines not available yet")

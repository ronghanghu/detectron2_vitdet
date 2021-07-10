# Copyright (c) Facebook, Inc. and its affiliates.
import os

from .ytvis import register_ytvis_instances

"""
This file registers the YTVIS dataset.
"""

__all__ = ["register_all_ytvis"]


# ==== Predefined datasets and splits for YTVIS ==========


_PREDEFINED_SPLITS_YTVIS = {
    "ytvis": {
        "ytvis_2019_train": (
            "memcache_manifold://fair_vision_data/tree/ytvos2019/train/JPEGImages",
            "ytvis2019/train.json",
        ),
        "ytvis_2019_valid": (
            "memcache_manifold://fair_vision_data/tree/ytvos2019/valid/JPEGImages",
            "ytvis2019/valid.json",
        ),
        "ytvis_2019_test": (
            "memcache_manifold://fair_vision_data/tree/ytvos2019/test/JPEGImages",
            "ytvis2019/test.json",
        ),
        "ytvis_sub_2019_train": (
            "memcache_manifold://fair_vision_data/tree/ytvos2019/train/JPEGImages",
            "train_subset.json",
        ),
        "ytvis_sub_2019_val": (
            "memcache_manifold://fair_vision_data/tree/ytvos2019/train/JPEGImages",
            "val_subset.json",
        ),
    }
}

JSON_ANNOTATIONS_DIR = "manifold://fair_vision_data/tree/detectron2/json_dataset_annotations/"
JSON_ANNOTATIONS_DIR_YTVIS_SUB = JSON_ANNOTATIONS_DIR + "ytvis2019/train_subset"


def register_all_ytvis():
    for _, splits_per_dataset in _PREDEFINED_SPLITS_YTVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            json_dir = (
                JSON_ANNOTATIONS_DIR_YTVIS_SUB
                if key.startswith("ytvis_sub")
                else JSON_ANNOTATIONS_DIR
            )
            register_ytvis_instances(key, {}, os.path.join(json_dir, json_file), image_root)


register_all_ytvis()

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
            "memcache_manifold://fair_vision_data/tree/ytvos2019/train",
            "ytvis2019/train.json",
        ),
        "ytvis_2019_valid": (
            "memcache_manifold://fair_vision_data/tree/ytvos2019/valid",
            "ytvis2019/valid.json",
        ),
        "ytvis_2019_test": (
            "memcache_manifold://fair_vision_data/tree/ytvos2019/test",
            "ytvis2019/test.json",
        ),
    }
}

JSON_ANNOTATIONS_DIR = "manifold://fair_vision_data/tree/detectron2/json_dataset_annotations/"


def register_all_ytvis():
    for _, splits_per_dataset in _PREDEFINED_SPLITS_YTVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_ytvis_instances(
                key, {}, os.path.join(JSON_ANNOTATIONS_DIR, json_file), image_root
            )


register_all_ytvis()

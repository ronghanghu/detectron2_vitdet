# -*- coding: utf-8 -*-


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly without having the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from .register_coco import register_coco_instances

# All coco categories, together with their nice-looking visualization colors
# fmt: off
COCO_CATEGORIES = [
    {"supercategory": "person", "color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"}, # noqa
    {"supercategory": "vehicle", "color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"}, # noqa
    {"supercategory": "vehicle", "color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"}, # noqa
    {"supercategory": "vehicle", "color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"}, # noqa
    {"supercategory": "vehicle", "color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"}, # noqa
    {"supercategory": "vehicle", "color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"}, # noqa
    {"supercategory": "vehicle", "color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"}, # noqa
    {"supercategory": "vehicle", "color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"}, # noqa
    {"supercategory": "vehicle", "color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"}, # noqa
    {"supercategory": "outdoor", "color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"}, # noqa
    {"supercategory": "outdoor", "color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"}, # noqa
    {"supercategory": "outdoor", "color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"}, # noqa
    {"supercategory": "outdoor", "color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"}, # noqa
    {"supercategory": "outdoor", "color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"}, # noqa
    {"supercategory": "animal", "color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"}, # noqa
    {"supercategory": "animal", "color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"}, # noqa
    {"supercategory": "animal", "color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"}, # noqa
    {"supercategory": "animal", "color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"}, # noqa
    {"supercategory": "animal", "color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"}, # noqa
    {"supercategory": "animal", "color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"}, # noqa
    {"supercategory": "animal", "color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"}, # noqa
    {"supercategory": "animal", "color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"}, # noqa
    {"supercategory": "animal", "color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"}, # noqa
    {"supercategory": "animal", "color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"}, # noqa
    {"supercategory": "accessory", "color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"}, # noqa
    {"supercategory": "accessory", "color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"}, # noqa
    {"supercategory": "accessory", "color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"}, # noqa
    {"supercategory": "accessory", "color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"}, # noqa
    {"supercategory": "accessory", "color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"}, # noqa
    {"supercategory": "sports", "color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"}, # noqa
    {"supercategory": "sports", "color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"}, # noqa
    {"supercategory": "sports", "color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"}, # noqa
    {"supercategory": "sports", "color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"}, # noqa
    {"supercategory": "sports", "color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"}, # noqa
    {"supercategory": "sports", "color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"}, # noqa
    {"supercategory": "sports", "color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"}, # noqa
    {"supercategory": "sports", "color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"}, # noqa
    {"supercategory": "sports", "color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"}, # noqa
    {"supercategory": "sports", "color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"}, # noqa
    {"supercategory": "kitchen", "color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"}, # noqa
    {"supercategory": "kitchen", "color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"}, # noqa
    {"supercategory": "kitchen", "color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"}, # noqa
    {"supercategory": "kitchen", "color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"}, # noqa
    {"supercategory": "kitchen", "color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"}, # noqa
    {"supercategory": "kitchen", "color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"}, # noqa
    {"supercategory": "kitchen", "color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"}, # noqa
    {"supercategory": "food", "color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"}, # noqa
    {"supercategory": "food", "color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"}, # noqa
    {"supercategory": "food", "color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"}, # noqa
    {"supercategory": "food", "color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"}, # noqa
    {"supercategory": "food", "color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"}, # noqa
    {"supercategory": "food", "color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"}, # noqa
    {"supercategory": "food", "color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"}, # noqa
    {"supercategory": "food", "color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"}, # noqa
    {"supercategory": "food", "color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"}, # noqa
    {"supercategory": "food", "color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"}, # noqa
    {"supercategory": "furniture", "color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"}, # noqa
    {"supercategory": "furniture", "color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"}, # noqa
    {"supercategory": "furniture", "color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"}, # noqa
    {"supercategory": "furniture", "color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"}, # noqa
    {"supercategory": "furniture", "color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"}, # noqa
    {"supercategory": "furniture", "color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"}, # noqa
    {"supercategory": "electronic", "color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"}, # noqa
    {"supercategory": "electronic", "color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"}, # noqa
    {"supercategory": "electronic", "color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"}, # noqa
    {"supercategory": "electronic", "color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"}, # noqa
    {"supercategory": "electronic", "color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"}, # noqa
    {"supercategory": "electronic", "color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"}, # noqa
    {"supercategory": "appliance", "color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"}, # noqa
    {"supercategory": "appliance", "color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"}, # noqa
    {"supercategory": "appliance", "color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"}, # noqa
    {"supercategory": "appliance", "color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"}, # noqa
    {"supercategory": "appliance", "color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"}, # noqa
    {"supercategory": "indoor", "color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"}, # noqa
    {"supercategory": "indoor", "color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"}, # noqa
    {"supercategory": "indoor", "color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"}, # noqa
    {"supercategory": "indoor", "color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"}, # noqa
    {"supercategory": "indoor", "color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"}, # noqa
    {"supercategory": "indoor", "color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"}, # noqa
    {"supercategory": "indoor", "color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"}, # noqa
    {"supercategory": "textile", "color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"}, # noqa
    {"supercategory": "textile", "color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"}, # noqa
    {"supercategory": "building", "color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"}, # noqa
    {"supercategory": "raw-material", "color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"}, # noqa
    {"supercategory": "furniture-stuff", "color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"}, # noqa
    {"supercategory": "textile", "color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"}, # noqa
    {"supercategory": "furniture-stuff", "color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"}, # noqa
    {"supercategory": "floor", "color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"}, # noqa
    {"supercategory": "plant", "color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"}, # noqa
    {"supercategory": "food-stuff", "color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"}, # noqa
    {"supercategory": "ground", "color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"}, # noqa
    {"supercategory": "building", "color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"}, # noqa
    {"supercategory": "furniture-stuff", "color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"}, # noqa
    {"supercategory": "furniture-stuff", "color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"}, # noqa
    {"supercategory": "structural", "color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"}, # noqa
    {"supercategory": "textile", "color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"}, # noqa
    {"supercategory": "ground", "color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"}, # noqa
    {"supercategory": "ground", "color": [106, 154, 176], "isthing": 0, "id": 145, "name": "playingfield"}, # noqa
    {"supercategory": "ground", "color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"}, # noqa
    {"supercategory": "water", "color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"}, # noqa
    {"supercategory": "ground", "color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"}, # noqa
    {"supercategory": "building", "color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"}, # noqa
    {"supercategory": "ground", "color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"}, # noqa
    {"supercategory": "water", "color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"}, # noqa
    {"supercategory": "furniture-stuff", "color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"}, # noqa
    {"supercategory": "ground", "color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"}, # noqa
    {"supercategory": "furniture-stuff", "color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"}, # noqa
    {"supercategory": "building", "color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"}, # noqa
    {"supercategory": "textile", "color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"}, # noqa
    {"supercategory": "wall", "color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"}, # noqa
    {"supercategory": "wall", "color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"}, # noqa
    {"supercategory": "wall", "color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"}, # noqa
    {"supercategory": "wall", "color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"}, # noqa
    {"supercategory": "water", "color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"}, # noqa
    {"supercategory": "window", "color": [183, 121, 142], "isthing": 0, "id": 180, "name": "window-blind"}, # noqa
    {"supercategory": "window", "color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"}, # noqa
    {"supercategory": "plant", "color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"}, # noqa
    {"supercategory": "structural", "color": [190, 153, 153], "isthing": 0, "id": 185, "name": "fence-merged"}, # noqa
    {"supercategory": "ceiling", "color": [146, 139, 141], "isthing": 0, "id": 186, "name": "ceiling-merged"}, # noqa
    {"supercategory": "sky", "color": [70, 130, 180], "isthing": 0, "id": 187, "name": "sky-other-merged"}, # noqa
    {"supercategory": "furniture-stuff", "color": [134, 199, 156], "isthing": 0, "id": 188, "name": "cabinet-merged"}, # noqa
    {"supercategory": "furniture-stuff", "color": [209, 226, 140], "isthing": 0, "id": 189, "name": "table-merged"}, # noqa
    {"supercategory": "floor", "color": [96, 36, 108], "isthing": 0, "id": 190, "name": "floor-other-merged"}, # noqa
    {"supercategory": "ground", "color": [96, 96, 96], "isthing": 0, "id": 191, "name": "pavement-merged"}, # noqa
    {"supercategory": "solid", "color": [64, 170, 64], "isthing": 0, "id": 192, "name": "mountain-merged"}, # noqa
    {"supercategory": "plant", "color": [152, 251, 152], "isthing": 0, "id": 193, "name": "grass-merged"}, # noqa
    {"supercategory": "ground", "color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"}, # noqa
    {"supercategory": "raw-material", "color": [206, 186, 171], "isthing": 0, "id": 195, "name": "paper-merged"}, # noqa
    {"supercategory": "food-stuff", "color": [152, 161, 64], "isthing": 0, "id": 196, "name": "food-other-merged"}, # noqa
    {"supercategory": "building", "color": [116, 112, 0], "isthing": 0, "id": 197, "name": "building-other-merged"}, # noqa
    {"supercategory": "solid", "color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"}, # noqa
    {"supercategory": "wall", "color": [102, 102, 156], "isthing": 0, "id": 199, "name": "wall-other-merged"}, # noqa
    {"supercategory": "textile", "color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"}, # noqa
]

COCO_PERSON_KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

# Pairs of keypoints that should be exchanged under horizontal flipping
COCO_PERSON_KEYPOINT_FLIP_MAP = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)

# rules for pairs of keypoints to draw a line between, and the line color to use.
KEYPOINT_CONNECTION_RULES = [
    # face
    ("left_ear", "left_eye", (102, 204, 255)),
    ("right_ear", "right_eye", (51, 153, 255)),
    ("left_eye", "nose", (102, 0, 204)),
    ("nose", "right_eye", (51, 102, 255)),
    # upper-body
    ("left_shoulder", "right_shoulder", (255, 128, 0)),
    ("left_shoulder", "left_elbow", (153, 255, 204)),
    ("right_shoulder", "right_elbow", (128, 229, 255)),
    ("left_elbow", "left_wrist", (153, 255, 153)),
    ("right_elbow", "right_wrist", (102, 255, 224)),
    # lower-body
    ("left_hip", "right_hip", (255, 102, 0)),
    ("left_hip", "left_knee", (255, 255, 77)),
    ("right_hip", "right_knee", (153, 255, 204)),
    ("left_knee", "left_ankle", (191, 255, 128)),
    ("right_knee", "right_ankle", (255, 195, 77)),
]

# fmt: on


def _get_coco_instances_meta():
    things_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    assert len(things_ids) == 80, len(things_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    dataset_id_to_contiguous_id = {k: i for i, k in enumerate(things_ids)}
    class_names = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "dataset_id_to_contiguous_id": dataset_id_to_contiguous_id,
        "class_names": class_names,
        "categories": COCO_CATEGORIES,
    }
    return ret


def _get_coco_panoptic_separated_meta():
    """
    Returns metadata for "separated" version of the panoptic segmentation dataset.
    """
    stuff_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    assert len(stuff_ids) == 53, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 53], used in models) to ids in the dataset (used for processing results)
    # The id 0 is mapped to an extra category "thing".
    stuff_contiguous_id_to_dataset_id = {i + 1: k for i, k in enumerate(stuff_ids)}
    # When converting COCO panoptic annotations to semantic annotations
    # We label the "thing" category to 0
    stuff_contiguous_id_to_dataset_id[0] = 255
    stuff_contiguous_id_to_dataset_id = stuff_contiguous_id_to_dataset_id

    # 54 names for COCO stuff categories (including "things")
    stuff_class_names = ["things"] + [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    stuff_class_names = stuff_class_names
    ret = {
        "stuff_contiguous_id_to_dataset_id": stuff_contiguous_id_to_dataset_id,
        "stuff_class_names": stuff_class_names,
        "categories": COCO_CATEGORIES,
    }
    ret.update(_get_coco_instances_meta())
    return ret


def _get_coco_densepose_meta():
    meta = {
        "class_names": ["person"],
        "densepose_transform_src": "detectron2://densepose/UV_symmetry_transforms.mat",
        "densepose_smpl_subdiv": "detectron2://densepose/SMPL_subdiv.mat",
        "densepose_smpl_subdiv_transform": "detectron2://densepose/SMPL_SUBDIV_TRANSFORM.mat",
        "densepose_pairwise_geod_dists": "detectron2://densepose/Pdist_matrix.mat",
    }
    return meta


def _get_builtin_metadata(dataset_name):
    if dataset_name == "coco":
        return _get_coco_instances_meta()
    if dataset_name == "coco_panoptic_separated":
        return _get_coco_panoptic_separated_meta()
    elif dataset_name == "coco_person":
        return {
            "class_names": ["person"],
            "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
            "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
            "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
        }
    elif dataset_name == "coco_densepose":
        return _get_coco_densepose_meta()
    elif dataset_name == "cityscapes":
        # We choose this order because it is consistent with our old json annotation files
        # TODO Perhaps switch to an order that's consistent with Cityscapes'
        # original label, when we don't need the legacy jsons any more.
        CITYSCAPES_THING_CLASS_NAMES = [
            "bicycle",
            "motorcycle",
            "rider",
            "train",
            "car",
            "person",
            "truck",
            "bus",
        ]
        return {"class_names": CITYSCAPES_THING_CLASS_NAMES}
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))


# ======================= Predefined datasets and splits ======================


# Some predefined datasets in COCO format
_PREDEFINED_SPLITS = {}
_PREDEFINED_SPLITS["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": (
        "coco/val2014",
        "detectron2://annotations/coco/instances_minival2014_100.json",
    ),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_val_100": (
        "coco/val2017",
        "detectron2://annotations/coco/instances_val2017_100.json",
    ),
}

_PREDEFINED_SPLITS["cityscapes"] = {
    # TODO understand what is "filtered"
    "cityscapes_fine_instanceonly_seg_train_cocostyle": (
        "cityscapes/images",
        "cityscapes/annotations/instancesonly_gtFine_train.json",
    ),
    "cityscapes_fine_instanceonly_seg_val_cocostyle": (
        "cityscapes/images",
        "cityscapes/annotations/instancesonly_filtered_gtFine_val.json",
    ),
    "cityscapes_fine_instanceonly_seg_test_cocostyle": (
        "cityscapes/images",
        "cityscapes/annotations/instancesonly_gtFine_test.json",
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
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "detectron2://annotations/coco/person_keypoints_val2017_100.json",
    ),
}

_PREDEFINED_SPLITS["coco_densepose"] = {
    "densepose_coco_2014_train": ("coco/train2014", "coco/annotations/densepose_train2014.json"),
    "densepose_coco_2014_minival": ("coco/val2014", "coco/annotations/densepose_minival2014.json"),
    "densepose_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/densepose_valminusminival2014.json",
    ),
}

for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS.items():
    for key, (image_root, json_file) in splits_per_dataset.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata(dataset_name),
            os.path.join("datasets", json_file) if "://" not in json_file else json_file,
            os.path.join("datasets", image_root),
        )

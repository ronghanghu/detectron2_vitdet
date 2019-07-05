import copy
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
from .coco import load_coco_json, load_sem_seg

"""
This file contains functions to register a COCO-format dataset to the DatasetCatalog.
"""

__all__ = ["register_coco_instances", "register_coco_panoptic_separated"]


def register_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.
            Currently only "class_names" is used, but "class_names" will also
            be loaded automatically from json, therefore you can leave it an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    if isinstance(metadata, str):
        # TODO for BC only. Remove this after a while
        metadata = {"dataset_name": metadata}
        logger = logging.getLogger(__name__)
        logger.warn(
            """
After D15247032, all metadata should be associated with dataset splits by
`register_coco_instances(name, metadata, ...)`.
`register_coco_instances(name, dataset_name, ...)` is deprecated since
"dataset_name" is no longer useful.  """
        )
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def register_coco_panoptic_separated(
    name, metadata, image_root, panoptic_root, panoptic_json, semantic_root, instances_json
):
    """
    Register a COCO panoptic segmentation dataset named `name`.
    The annotations in this registered dataset will contain both instance annotations and
    semantic annotations, each with its own contiguous ids. Hence it's called "separated".

    The instance annotations directly comes from polygons in the COCO
    instances annotation task, rather than from the masks in the COCO panoptic annotations.
    We now use the polygons because it is what the PanopticFPN paper uses.
    We may later need to add an option to use masks.
    The two format have small differences:
    Polygons in the instance annotations may have overlaps.
    The mask annotations are produced by labeling the depth ordering of overlapped polygons.

    The semantic annotations are converted from panoptic annotations, where
    all things are assigned a semantic id of 0.
    All semantic categories will therefore have ids in contiguous range [0, #stuff_categories).

    This function will also register a pure semantic segmentation dataset
    named `name + '_stuffonly'`.

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (str): extra metadata associated with this dataset
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images
        panoptic_json (str): path to the json panoptic annotation file
        semantic_root (str): directory which contains all the ground truth segmentation annotations.
            It can be converted from `panoptic_root` by scripts in
            https://github.com/cocodataset/panopticapi/tree/master/converters
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name + "_separated"
    DatasetCatalog.register(
        panoptic_name,
        lambda: merge_to_panoptic(
            load_coco_json(instances_json, image_root, panoptic_name),
            load_sem_seg(semantic_root, image_root),
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        semantic_root=semantic_root,
        json_file=instances_json,  # TODO rename
        evaluator_type="coco_panoptic_seg",
        **metadata
    )

    semantic_name = name + "_stuffonly"
    DatasetCatalog.register(semantic_name, lambda: load_sem_seg(semantic_root, image_root))
    MetadataCatalog.get(semantic_name).set(
        semantic_root=semantic_root,
        image_root=image_root,
        evaluator_type="semantic_seg",
        **metadata
    )


def merge_to_panoptic(detection_dicts, semantic_segmentation_dicts):
    """
    Create dataset dicts for panoptic segmentation, by
    merging two dicts using "file_name" field to match their entries.

    Args:
        detection_dicts (list[dict]): lists of dicts for object detection or instance segmentation.
        semantic_segmentation_dicts (list[dict]): lists of dicts for semantic segmentation.

    Returns:
        list[dict] (one per input image): Each dict contains all (key, value) pairs from dicts in
            both detection_dicts and semantic_segmentation_dicts that correspond to the same image.
            The function assumes that the same key in different dicts has the same value.
    """
    results = []
    semseg_file_to_entry = {x["file_name"]: x for x in semantic_segmentation_dicts}

    for det_dict in detection_dicts:
        dic = copy.copy(det_dict)
        dic.update(semseg_file_to_entry[dic["file_name"]])
        results.append(dic)
    return results

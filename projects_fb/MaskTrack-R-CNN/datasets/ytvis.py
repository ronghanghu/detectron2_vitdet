# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import logging
import os
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

"""
This file contains functions to parse YTVIS-format annotations into dicts in the
"Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_ytvis_json", "register_ytvis_instances"]


def load_ytvis_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with YTVIS's instances annotation format.
    Currently supports instance segmentation annotations.

    Args:
        json_file (str): full path to the json file in YTVIS instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., ytvis_2019_train).
            When provided, this function will also do the following:
            * Put "thing_classes" into the metadata associated with this dataset.
            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.

    Returns:
        list[dict]: list of dicts, one for each video clip. Each dict has keys "length" (number
        of frames in video), "id" (video id), and "frames" (list[dict]). Each dict in "frames"
        has keys related to the specific frame like "file_name", "height", "width", "annotations".
        See below for specific format of "annotations". Each dict in "frames" follows the
        standard Detectron2 dataset dict format for images (See
        </https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html>).

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools_ytvos.ytvos import YTVOS

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ytvos_api = YTVOS(json_file)
    if timer.seconds() > 1:
        logger.info(f"Loading {json_file} takes {timer.seconds():.2f} seconds")

    # set meta data
    if dataset_name is not None:
        meta = get_ytvis_instances_meta(ytvos_api, dataset_name)
        MetadataCatalog.get(dataset_name).set(**meta)

    # sort indices for reproducible results
    vid_ids = sorted(ytvos_api.getVidIds())
    vids = ytvos_api.loadVids(vid_ids)
    logger.info(f"Loaded {len(vids)} videos in YTVIS format from {json_file}")
    # vids is a list of dicts, each looks something like:
    # {'width': 1280,
    #  'height': 720,
    #  'length': 20,
    #  'date_captured': '2019-04-11 00:55:41.903902',
    #  'file_names': ['0043f083b5/00000.jpg', '0043f083b5/00005.jpg', ...],
    #  'id': 1}
    ann_keys = ["iscrowd", "category_id"] + (extra_annotation_keys or [])
    num_instances_without_valid_segmentation = 0

    dataset_dicts = []
    for vid_dict in vids:
        video_record = {k: vid_dict[k] for k in ("length", "id")}
        video_id = vid_dict["id"]
        annotation_ids = ytvos_api.getAnnIds(vidIds=[video_id])
        annotations = ytvos_api.loadAnns(annotation_ids)
        # annotations is a list of dicts, where each dict is an
        # annotation record for an object over the video. Example of annoations[0]:
        # {'width': 1280,
        #  'height': 720,
        #  'length': 1,
        #  'category_id': 6,
        #  'segmentations': [None, None, ..., {
        #   'counts': [],
        #   'size': [720, 1280]}, ...],
        #  'bboxes': [None, None, ..., [1153.0, 372.0, 21.0, 54], ...],
        #  'video_id': 1,
        #  'iscrowd': 0,
        #  'id': 1,
        #  'areas': [None, None, ..., 910, ...]}

        video_record["frames"] = []
        for frame_idx, file_name in enumerate(vid_dict["file_names"]):
            # create a record for every video frame
            record = {}
            record["file_name"] = os.path.join(image_root, file_name)
            record["height"] = vid_dict["height"]
            record["width"] = vid_dict["width"]
            record["video_id"] = video_id

            # annotation objects
            objs = []
            for anno in annotations:
                assert anno["video_id"] == video_id
                obj = {key: anno[key] for key in ann_keys if key in anno}

                # segmentations
                segm_list = anno.get("segmentations")
                segm = segm_list[frame_idx] if segm_list else None
                if segm:
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                    else:
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue  # ignore this instance
                    obj["segmentation"] = segm

                # bboxes
                bboxes_list = anno["bboxes"]
                bbox = bboxes_list[frame_idx]

                # Checks if object is still in frame
                if bbox is not None:
                    obj["bbox"] = bboxes_list[frame_idx]
                    obj["bbox_mode"] = BoxMode.XYWH_ABS
                    obj["category_id"] = anno["category_id"] - 1
                    obj["instance_id"] = anno["id"]
                    objs.append(obj)

            record["annotations"] = objs
            video_record["frames"].append(record)
        dataset_dicts.append(video_record)

    logger.info(f"Loaded {len(dataset_dicts)} videos in YTVIS format from {json_file}")
    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            f"Filtered out {num_instances_without_valid_segmentation}"
            "instances without valid segmentation. "
            "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def get_ytvis_instances_meta(ytvos_api, dataset_name):
    """
    Loads YTVIS metadata.

    Args:
        dataset_name: YTVIS dataset name
        ytvos_api: Interface for accessing YTVIS dataset

    Returns:
        dict: YTVIS metadata with keys:
    """
    cat_ids = sorted(ytvos_api.getCatIds())
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    categories = sorted(ytvos_api.loadCats(cat_ids), key=lambda x: x["id"])
    thing_classes = [c["name"] for c in categories]
    meta = {"thing_classes": thing_classes}
    return meta


def register_ytvis_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in YTVIS's json annotation format.

    Args:
        name (str): a name that identifies the dataset, e.g. "ytvis_2019_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_ytvis_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="ytvis", **metadata
    )

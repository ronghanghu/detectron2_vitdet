# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import numpy as np
import torch

from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

"""
This file contains the mapping that's applied to the YTVIS dataset.
"""

__all__ = ["YTVISDatasetMapper"]


class YTVISDatasetMapper(DatasetMapper):
    """
    A callable which takes a dataset dict from YTVIS,
    and maps it into a format used by MaskTrackRCNN.
    """

    def add_frame(self, frame_prefix, input_dict):
        """
        Args:
            frame_prefix (str): Key for storing frame level information. Used
                for specifying reference frame if set to value "ref_".
            input_dict (dict): Input dict to model
        """
        image = utils.read_image(input_dict[frame_prefix + "file_name"], format=self.image_format)
        utils.check_image_size(input_dict, image)

        aug_input = T.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        input_dict[frame_prefix + "image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            input_dict.pop(frame_prefix + "annotations", None)
            return input_dict

        for anno in input_dict[frame_prefix + "annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)

        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=None
            )
            for obj in input_dict.pop(frame_prefix + "annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )
        instances, filters = utils.filter_empty_instances(instances, return_mask=True)
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        input_dict[frame_prefix + "instances"] = instances
        if frame_prefix == "":
            input_dict["gt_ids"] = [i for (i, v) in zip(input_dict["gt_ids"], filters) if v]
        else:
            ref_ids = [i for (i, v) in zip(input_dict["ref_ids"], filters) if v]
            input_dict["gt_pids"] = [
                ref_ids.index(i) + 1 if i in ref_ids else 0 for i in input_dict["gt_ids"]
            ]
        return input_dict

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video clip.
        Returns:
            dict: in a format that a MaskTrackRCNN model accepts
        """
        input_dict = {}
        if self.is_train:
            frame_indices = np.random.choice(len(dataset_dict["frames"]), 2, replace=False)
            gt_frame_dict = dataset_dict["frames"][frame_indices[0]]
            ref_frame_dict = dataset_dict["frames"][frame_indices[1]]

            input_dict = copy.deepcopy(gt_frame_dict)
            input_dict["gt_ids"] = [ann["instance_id"] for ann in gt_frame_dict["annotations"]]
            input_dict["ref_ids"] = [ann["instance_id"] for ann in ref_frame_dict["annotations"]]
            input_dict["gt_pids"] = [
                input_dict["ref_ids"].index(i) + 1 if i in input_dict["ref_ids"] else 0
                for i in input_dict["gt_ids"]
            ]

            self.add_frame("", input_dict)
            input_dict["ref_file_name"] = ref_frame_dict["file_name"]
            input_dict["ref_annotations"] = copy.deepcopy(ref_frame_dict["annotations"])
            self.add_frame("ref_", input_dict)
            input_dict["instances"].gt_pids = torch.tensor(
                input_dict.pop("gt_pids"), dtype=torch.int64
            )
        else:
            assert len(dataset_dict["frames"]) == 1  # inference works on one frame at a time
            frame_dict = dataset_dict["frames"][0]
            input_dict = copy.deepcopy(frame_dict)
            input_dict["is_first"] = frame_dict["frame_idx"] == 0
            self.add_frame("", input_dict)
        return input_dict

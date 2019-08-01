import copy
import numpy as np
import torch
from PIL import Image

from detectron2.structures import DensePoseDataRelative, DensePoseTransformData
from detectron2.utils.file_io import PathManager

from . import detection_utils as utils
from . import transforms as T
from .catalog import MetadataCatalog

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic.

    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.densepose_on   = cfg.MODEL.DENSEPOSE_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        # TODO: this is temporary. We're aiming to gradually refactor
        # densepose to become "user code"
        # instead of injecting it inside detectron2 in many places.
        if self.densepose_on:
            densepose_transform_srcs = [
                MetadataCatalog.get(ds).densepose_transform_src
                for ds in cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
            ]
            assert len(densepose_transform_srcs) > 0
            # TODO: check that DensePose transformation data is the same for
            # all the datasets. Otherwise one would have to pass DB ID with
            # each entry to select proper transformation data. For now, since
            # all DensePose annotated data uses the same data semantics, we
            # omit this check.
            densepose_transform_data_fpath = PathManager.get_file_name(densepose_transform_srcs[0])
            self.densepose_transform_data = DensePoseTransformData.load(
                densepose_transform_data_fpath
            )

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            # USER: Don't call transpose_densepose if you don't need
            annos = [
                self._transform_densepose(
                    utils.transform_instance_annotations(
                        obj,
                        transforms,
                        image_shape,
                        keypoint_hflip_indices=self.keypoint_hflip_indices,
                    ),
                    transforms,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            targets = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["targets"] = targets[targets.gt_boxes.nonempty()]

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = Image.open(dataset_dict.pop("sem_seg_file_name"))
            sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg_gt"] = sem_seg_gt
        # TODO find a better way to load panoptic data than using the "separate" dataset
        return dataset_dict

    # TODO Keep it in this class for now since we'll move it to a separate directory
    def _transform_densepose(self, annotation, transforms):
        if not self.densepose_on:
            return annotation

        # Handle densepose annotations
        is_valid, reason_not_valid = DensePoseDataRelative.validate_annotation(annotation)
        if is_valid:
            densepose_data = DensePoseDataRelative(annotation, cleanup=True)
            densepose_data.apply_transform(transforms, self.densepose_transform_data)
            annotation["densepose"] = densepose_data
        else:
            # logger = logging.getLogger(__name__)
            # logger.debug("Could not load DensePose annotation: {}".format(reason_not_valid))
            DensePoseDataRelative.cleanup_annotation(annotation)
            # NOTE: annotations for certain instances may be unavailable.
            # This is accepted by the DensePostList data structure.
            annotation["densepose"] = None
        return annotation

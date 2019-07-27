import copy
import logging
import numpy as np
import torch
from PIL import Image

from detectron2.structures import (
    Boxes,
    BoxMode,
    DensePoseDataRelative,
    DensePoseList,
    DensePoseTransformData,
    Instances,
    Keypoints,
    PolygonMasks,
)
from detectron2.utils.file_io import PathManager

from . import transforms as T
from .catalog import MetadataCatalog

"""
This file contains the default transformation that's applied to "dataset dicts".
It needs to be made easier to customize.
"""

__all__ = ["DetectionTransform"]


class SizeMismatchError(ValueError):
    """
    When loaded image has difference width/height compared with annoation.
    """


def annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of annotations, one per instance.
        image_size (tuple): height, width

    Returns:
        Instances: It will contains fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        masks = [obj["segmentation"] for obj in annos]
        masks = PolygonMasks(masks)
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    if len(annos) and "densepose" in annos[0]:
        gt_densepose = [obj["densepose"] for obj in annos]
        target.gt_densepose = DensePoseList(gt_densepose, boxes, image_size)

    target = target[boxes.nonempty()]
    return target


class DetectionTransform:
    """
    A callable which takes a dict produced by the detection dataset, and applies transformations,
    including image resizing and flipping. The transformation parameters are parsed from cfg file
    and depending on the is_train condition.

    Note that for our existing models, mean/std normalization is done by the model instead of here.
    """

    def __init__(self, cfg, is_train=True):
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        if sample_style == "range":
            assert (
                len(min_size) == 2
            ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

        self.img_format = cfg.INPUT.FORMAT

        logger = logging.getLogger(__name__)
        self.tfm_gens = []
        if not min_size == 0:  # set to zero to disable resize
            self.tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
        if is_train:
            self.tfm_gens.append(T.RandomFlip())
            logger.info("TransformGens used in training: " + str(self.tfm_gens))

        self.mask_on = cfg.MODEL.MASK_ON
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.densepose_on = cfg.MODEL.DENSEPOSE_ON
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_flip_indices = _create_flip_indices(cfg)
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

        self.is_train = is_train

        if cfg.MODEL.LOAD_PROPOSALS:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )

    def __call__(self, dataset_dict):
        """
        Transform the dataset_dict according to the configured transformations.

        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a new dict that's going to be processed by the model.
                It currently does the following:
                1. Read the image from "file_name"
                2. Transform the image and annotations
                3. Prepare the annotations to :class:`Instances`
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = self._read_image(dataset_dict, format=self.img_format)
        DetectionTransform.check_image_size(dataset_dict, image)
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day
        dataset_dict["image"] = image

        self.transform_precomputed_proposals(dataset_dict, image_shape, transforms)

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            annos = [
                self.transform_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # Should not be empty during training
            dataset_dict["targets"] = annotations_to_instances(annos, image_shape)
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = Image.open(dataset_dict.pop("sem_seg_file_name"))
            sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg_gt"] = sem_seg_gt
        return dataset_dict

    def transform_precomputed_proposals(self, dataset_dict, image_shape, transforms):
        """
        Apply transformations to the precomputed proposals in dataset_dict.

        Args:
            dataset_dict (dict): a dict read from the dataset, possibly
                contains fields "proposal_boxes", "proposal_objectness_logits", "proposal_bbox_mode"

        The input dict is modified in-place, with abovementioned keys removed. A new
        key "proposals" will be added. Its value is an `Instances`
        object which contains the transformed proposals in its field
        "proposal_boxes" and "objectness_logits".
        """
        if "proposal_boxes" in dataset_dict:
            # Tranform proposal boxes
            boxes = transforms.apply_box(
                BoxMode.convert(
                    dataset_dict.pop("proposal_boxes"),
                    dataset_dict.pop("proposal_bbox_mode"),
                    BoxMode.XYXY_ABS,
                )
            )
            boxes = Boxes(boxes)
            objectness_logits = torch.as_tensor(
                dataset_dict.pop("proposal_objectness_logits").astype("float32")
            )

            boxes.clip(image_shape)
            keep = boxes.nonempty(threshold=self.min_box_side_len)
            boxes = boxes[keep]
            objectness_logits = objectness_logits[keep]

            proposals = Instances(image_shape)
            proposals.proposal_boxes = boxes[: self.proposal_topk]
            proposals.objectness_logits = objectness_logits[: self.proposal_topk]
            dataset_dict["proposals"] = proposals

    @staticmethod
    def check_image_size(dataset_dict, image):
        if "width" in dataset_dict or "height" in dataset_dict:
            image_wh = (image.shape[1], image.shape[0])
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            if not image_wh == expected_wh:
                raise SizeMismatchError(
                    "mismatch (W,H), got {}, expect {}".format(image_wh, expected_wh)
                )

    def _read_image(self, dataset_dict, format=None):
        """
        Read an image into the given format.

        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
            format (dict): one of the supported image modes in PIL, or "BGR"

        Returns:
            image (np.ndarray): an HWC image
        """
        image = Image.open(dataset_dict["file_name"])

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image

    def transform_annotations(self, annotation, transforms, image_size):
        """
        Apply image transformations to the instance annotations.

        After this method, the box mode will be set to XYXY_ABS.
        """
        bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # Note that bbox is 1d (per-instance bounding box)
        annotation["bbox"] = transforms.apply_box([bbox])[0]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

        # each instance contains 1 or more polygons
        if self.mask_on and "segmentation" in annotation:
            annotation["segmentation"] = [
                transforms.apply_coords(np.asarray(p).reshape(-1, 2)).reshape(-1)
                for p in annotation["segmentation"]
            ]
        else:
            annotation.pop("segmentation", None)

        if self.keypoint_on and "keypoints" in annotation:
            _, image_width = image_size
            keypoints = self._process_keypoints(annotation["keypoints"], transforms, image_width)
            annotation["keypoints"] = keypoints
        else:
            annotation.pop("keypoints", None)

        if self.densepose_on:
            is_valid, reason_not_valid = DensePoseDataRelative.validate_annotation(annotation)
            if is_valid:
                densepose_data = DensePoseDataRelative(annotation, cleanup=True)
                densepose_data.apply_transform(transforms, self.densepose_transform_data)
                annotation["densepose"] = densepose_data
            else:
                # logger = logging.getLogger(__name__)
                # logger.debug("Could not load DensePose annotation: {}".format(reason_not_valid))
                DensePoseDataRelative.cleanup_annotation(annotation)
                annotation["densepose"] = None

        return annotation

    def _process_keypoints(self, keypoints, transforms, image_width):
        # (N*3,) -> (N, 3)
        keypoints = np.asarray(keypoints, dtype="float64").reshape(-1, 3)
        keypoints[:, :2] = transforms.apply_coords(keypoints[:, :2])

        # This assumes that HorizFlipTransform is the only one that does flip
        do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1

        # Alternative way: check if probe points was horizontally flipped.
        # probe = np.asarray([[0.0, 0.0], [image_width, 0.0]])
        # probe_aug = transforms.apply_coords(probe.copy())
        # do_hflip = np.sign(probe[1][0] - probe[0][0]) != np.sign(probe_aug[1][0] - probe_aug[0][0])  # noqa

        # If flipped, swap each keypoint with its opposite-handed equivalent
        if do_hflip:
            keypoints = keypoints[self.keypoint_flip_indices, :]

        # Maintain COCO convention that if visibility == 0, then x, y = 0
        keypoints[keypoints[:, 2] == 0] = 0
        return keypoints


def _create_flip_indices(cfg):
    names_per_dataset = [MetadataCatalog.get(ds).keypoint_names for ds in cfg.DATASETS.TRAIN]
    flip_maps_per_dataset = [MetadataCatalog.get(ds).keypoint_flip_map for ds in cfg.DATASETS.TRAIN]

    logger = logging.getLogger(__name__)

    def _check_consistent(name, entries_per_dataset):
        for idx, entry in enumerate(entries_per_dataset):
            if entry != entries_per_dataset[0]:
                logger.error(
                    "{} for dataset {} is {}".format(name, cfg.DATASETS.TRAIN[idx], str(entry))
                )
                logger.error(
                    "{} for dataset {} is {}".format(
                        name, cfg.DATASETS.TRAIN[0], str(entries_per_dataset[0])
                    )
                )
                raise ValueError("Training on several datasets with different '{}'!".format(name))

    _check_consistent("keypoint_names", names_per_dataset)
    _check_consistent("keypoint_flip_map", flip_maps_per_dataset)

    names = names_per_dataset[0]
    flip_map = dict(flip_maps_per_dataset[0])
    flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in flip_map else flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return np.asarray(flip_indices)

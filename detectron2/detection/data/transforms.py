import copy
import logging
import numpy as np
import torch
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.data.transforms import Flip, ImageTransformers, ResizeShortestEdge
from detectron2.structures import Boxes, BoxMode, Instances, Keypoints, PolygonMasks

__all__ = ["DetectionTransform"]


def annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models, from annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of annotations, one per instance.
        image_size (tuple): height, width

    Returns:
        Instances:
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

        tfms = []
        if not min_size == 0:  # set to zero to disable resize
            tfms.append(ResizeShortestEdge(min_size, max_size, sample_style))
        if is_train:
            tfms.append(Flip(horiz=True))
        self.tfms = ImageTransformers(tfms)

        self.mask_on = cfg.MODEL.MASK_ON
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_flip_indices = _create_flip_indices(cfg)

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
        image = self._read_image(dataset_dict["file_name"], format=self.img_format)
        image, tfm_params = self.tfms.transform_image_get_params(image)

        # PIL squeezes out the channel dimension for "L", so make it HWC
        if self.img_format == "L":
            image = np.expand_dims(image, -1)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day
        dataset_dict["image"] = image

        if "proposal_boxes" in dataset_dict:
            # Tranform proposal boxes
            boxes = self.transform_bbox(
                dataset_dict.pop("proposal_boxes"), dataset_dict["proposal_bbox_mode"], tfm_params
            )
            dataset_dict["proposal_bbox_mode"] = BoxMode.XYXY_ABS
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

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            annos = [
                self.transform_annotations(obj, tfm_params, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            targets = annotations_to_instances(annos, image_shape)
            # Should not be empty during training
            dataset_dict["targets"] = targets
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = Image.open(dataset_dict.pop("sem_seg_file_name"))
            sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = self.tfms.transform_segmentation(sem_seg_gt, tfm_params)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg_gt"] = sem_seg_gt
        return dataset_dict

    @staticmethod
    def _read_image(file_name, format=None):
        """
        Read an image into the given format.

        Args:
            file_name (str):
            format (str): one of the supported image modes in PIL, or "BGR"

        Returns:
            image (np.ndarray)
        """
        image = Image.open(file_name)

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
        return image

    def transform_annotations(self, annotation, tfm_params, image_size):
        """
        Apply image transformations to the annotations.

        After this method, the box mode will be set to XYXY_ABS.
        """
        annotation["bbox"] = self.transform_bbox(
            annotation["bbox"], annotation["bbox_mode"], tfm_params
        )
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

        # each instance contains 1 or more polygons
        if self.mask_on and "segmentation" in annotation:
            annotation["segmentation"] = [
                self.tfms.transform_coords(np.asarray(p).reshape(-1, 2), tfm_params).reshape(-1)
                for p in annotation["segmentation"]
            ]
        else:
            annotation.pop("segmentation", None)

        if self.keypoint_on and "keypoints" in annotation:
            _, image_width = image_size
            keypoints = self._process_keypoints(annotation["keypoints"], tfm_params, image_width)
            annotation["keypoints"] = keypoints
        else:
            annotation.pop("keypoints", None)

        return annotation

    def transform_bbox(self, boxes, box_mode, tfm_params):
        """
        Apply image transformations to the boxes.
        The box modes of output boxes would be XYXY_ABS.
        """
        # First convert input boxes to XXXY_ABS mode.
        convert_boxes = BoxMode.convert(boxes, box_mode, BoxMode.XYXY_ABS)
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1])
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.array(convert_boxes).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.tfms.transform_coords(coords, tfm_params)
        coords = coords.reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)

        if not isinstance(boxes, np.ndarray):
            return type(boxes)(trans_boxes.flatten())

        return trans_boxes

    def _process_keypoints(self, keypoints, tfm_params, image_width):
        # (N*3,) -> (N, 3)
        keypoints = np.asarray(keypoints, dtype="float64").reshape(-1, 3)
        keypoints[:, :2] = self.tfms.transform_coords(keypoints[:, :2], tfm_params)

        # Check if the keypoints were horizontally flipped
        # If so, swap each keypoint with its opposite-handed equivalent
        probe = np.asarray([[0.0, 0.0], [image_width, 0.0]])
        probe_aug = self.tfms.transform_coords(probe.copy(), tfm_params)

        if np.sign(probe[1][0] - probe[0][0]) != np.sign(probe_aug[1][0] - probe_aug[0][0]):
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

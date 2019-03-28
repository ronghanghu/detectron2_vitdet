import copy
import numpy as np
import torch
from PIL import Image

from detectron2.data.transforms import Flip, ImageTransformers, Normalize, ResizeShortestEdge
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
    classes = torch.tensor(classes)
    target.gt_classes = classes

    masks = [obj["segmentation"] for obj in annos]
    masks = PolygonMasks(masks)
    target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    target = target[boxes.nonempty()]
    return target


# TODO this should be more accessible to users and be customizable
class DetectionTransform:
    """
    A callable which takes a dict produced by the detection dataset, and applies transformations.
    """

    def __init__(self, cfg, is_train=True):
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"

        # in testing, no random sample happens for now
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        if sample_style == "range":
            assert (
                len(min_size) == 2
            ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

        self.to_bgr = cfg.INPUT.BGR
        tfms = [ResizeShortestEdge(min_size, max_size, sample_style)]
        if is_train:
            tfms.append(Flip(horiz=True))
        tfms.append(Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD))
        self.tfms = ImageTransformers(tfms)
        self.is_train = is_train
        self.keypoint_flip_indices = _create_flip_indices(cfg)
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON

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
        image = Image.open(dataset_dict.pop("file_name")).convert("RGB")
        image = np.asarray(image, dtype="uint8")
        if self.to_bgr:
            image = image[:, :, ::-1]

        image, tfm_params = self.tfms.transform_image_get_params(image)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day
        dataset_dict["image"] = image

        if not self.is_train:
            del dataset_dict["annotations"]
            return dataset_dict

        annos = [
            self.transform_annotations(obj, tfm_params, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        targets = annotations_to_instances(annos, image_shape)
        # should not be empty during training
        dataset_dict["targets"] = targets
        return dataset_dict

    def transform_annotations(self, annotation, tfm_params, image_size):
        x, y, w, h = annotation["bbox"]
        coords = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype="float32")
        coords = self.tfms.transform_coords(coords, tfm_params)
        minxy = coords.min(axis=0)
        wh = coords.max(axis=0) - minxy
        annotation["bbox"] = (minxy[0], minxy[1], wh[0], wh[1])

        # each instance contains 1 or more polygons
        annotation["segmentation"] = [
            self.tfms.transform_coords(np.asarray(p).reshape(-1, 2), tfm_params).reshape(-1)
            for p in annotation["segmentation"]
        ]

        if self.keypoint_on and "keypoints" in annotation:
            _, image_width = image_size
            keypoints = self._process_keypoints(annotation["keypoints"], tfm_params, image_width)
            annotation["keypoints"] = keypoints

        return annotation

    def _process_keypoints(self, keypoints, tfm_params, image_width):
        # (N*3,) -> (N, 3)
        keypoints = np.asarray(keypoints).reshape(-1, 3)
        self.tfms.transform_coords(keypoints[:, :2], tfm_params)

        # Check if the keypoints were horizontally flipped
        # If so, swap each keypoint with its opposite-handed equivalent
        probe = np.asarray([[0.0, 0.0], [image_width, 0.0]])
        probe_aug = self.tfms.transform_coords(probe.copy(), tfm_params)

        if np.sign(probe[1][0] - probe[0][0]) != np.sign(probe_aug[1][0] - probe_aug[0][0]):
            keypoints = keypoints[self.keypoint_flip_indices, :]

        # Maintain COCO convention that if visibility == 0, then x, y = 0
        inds = keypoints[:, 2] == 0
        keypoints[inds] = 0
        return keypoints


def _create_flip_indices(cfg):
    names = cfg.MODEL.ROI_KEYPOINT_HEAD.KEYPOINT_NAMES
    flip_map = dict(cfg.MODEL.ROI_KEYPOINT_HEAD.KEYPOINT_FLIP_MAP)
    flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in flip_map else flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return np.asarray(flip_indices)

import numpy as np
import copy
from PIL import Image

from detectron2.data.transforms import Flip, ImageTransformers, Normalize, ResizeShortestEdge

__all__ = ["DetectionTransform"]


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
        The dataset_dict is modified in place.

        Args:
            dataset_dict (dict): A COCO-format annotation dict of one image.

        Returns:
            dict: the in-place modified dataset_dict where the annotations are
                replaced by transformed annotations (according to the configured
                transformations) and a new key is inserted:
                    image: the transformed image as a uint8 numpy array
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = Image.open(dataset_dict["file_name"]).convert("RGB")
        image = np.asarray(image, dtype="uint8")
        if self.to_bgr:
            image = image[:, :, ::-1]

        image, tfm_params = self.tfms.transform_image_get_params(image)
        dataset_dict["image"] = image

        if not self.is_train:
            del dataset_dict["annotations"]
            return dataset_dict

        annos = [
            self.map_instance(obj, tfm_params, image.shape[:2])
            for obj in dataset_dict["annotations"]
            if obj.get("iscrowd", 0) == 0
        ]
        # should not be empty during training
        dataset_dict["annotations"] = annos
        return dataset_dict

    def map_instance(self, annotation, tfm_params, image_size):
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

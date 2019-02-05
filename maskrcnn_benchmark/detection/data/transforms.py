import numpy as np
from PIL import Image

from maskrcnn_benchmark.data.transforms import (
    Flip,
    ImageTransformers,
    Normalize,
    ResizeShortestEdge,
)


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

    def __call__(self, dataset_dict):
        """
        Transform the dataset_dict according to the configured transformations.
        The dataset_dict is modified in place.

        Args:
            dataset_dict (dict): A dict with the following keys:
                file_name: str with the path to the image for this dataset entry
                annotations: list of dicts where each dict has the following keys:
                    bbox: bounding box for an object given by x0, y0, width, height
                    segmentation: polygon vertices in the COCO dataset format
                    is_crowd (optional): bool indicating if the object is a crowd
                        region or not

        Returns:
            dict: the in-place modified dataset_dict where the annotations are
                replaced by transformed annotations (according to the configured
                transformations) and the following keys are inserted:
                    image: the transformed image as a uint8 numpy array
                    transformed_width: the width of the transformed image
                    transformed_height: the height of the transformed image
        """
        image = Image.open(dataset_dict["file_name"]).convert("RGB")
        image = np.asarray(image, dtype="uint8")
        if self.to_bgr:
            image = image[:, :, ::-1]

        image, tfm_params = self.tfms.transform_image_get_params(image)
        dataset_dict["image"] = image
        dataset_dict["transformed_width"] = image.shape[1]
        dataset_dict["transformed_height"] = image.shape[0]

        if not self.is_train:
            del dataset_dict["annotations"]
            return dataset_dict

        annos = [
            self.map_instance(obj, tfm_params)
            for obj in dataset_dict["annotations"]
            if obj.get("iscrowd", 0) == 0
        ]
        # should not be empty during training
        dataset_dict["annotations"] = annos
        return dataset_dict

    def map_instance(self, annotation, tfm_params):
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
        return annotation

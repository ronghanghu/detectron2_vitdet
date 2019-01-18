import numpy as np
from copy import deepcopy
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

    def __call__(self, roidb):
        roidb = deepcopy(roidb)
        image = Image.open(roidb["file_name"]).convert("RGB")
        image = np.asarray(image, dtype="uint8")
        if self.to_bgr:
            image = image[:, :, ::-1]

        image, tfm_params = self.tfms.transform_image_get_params(image)
        roidb["image"] = image
        roidb["original_width"] = roidb["width"]
        roidb["original_height"] = roidb["height"]
        roidb["width"] = image.shape[1]
        roidb["height"] = image.shape[0]

        if not self.is_train:
            del roidb["annotations"]
            return roidb

        annos = [
            self.map_instance(obj, tfm_params)
            for obj in roidb["annotations"]
            if obj.get("iscrowd", 0) == 0
        ]
        # should not be empty during training
        roidb["annotations"] = annos
        return roidb

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

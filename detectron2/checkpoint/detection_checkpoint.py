import logging
import numpy as np
import pickle
from borc.common.file_io import PathManager

from .c2_model_loading import align_and_update_state_dicts
from .checkpoint import Checkpointer


def _convert_background_class(model):
    """
    Automatically convert the model for breakage in D14880072.
    """
    # TODO This is temporary. Remove this function after a while

    keys = model.keys()
    bbox_pred = [k for k in keys if "box_predictor.bbox_pred.weight" in k]
    cls_score = [k for k in keys if "box_predictor.cls_score.weight" in k]

    if len(bbox_pred) == 0:
        return model

    # assume you only have one mask r-cnn ..
    assert len(bbox_pred) == len(cls_score) and len(bbox_pred) == 1
    bbox_pred, cls_score = bbox_pred[0], cls_score[0]

    num_class_bbox = model[bbox_pred].shape[0] / 4
    num_class_cls = model[cls_score].shape[0]

    if num_class_bbox == num_class_cls:
        need_conversion = True  # this model is trained before D14880072.
    else:
        assert num_class_cls == num_class_bbox + 1
        need_conversion = False

    if not need_conversion:
        return model

    logger = logging.getLogger(__name__)
    logger.warning(
        "Your weights are in an old format! Please see D14880072 and convert your weights!"
    )
    logger.warning("Now attempting to automatically convert the weights for you ...")

    for k in keys:
        if "roi_heads.box_predictor.bbox_pred." in k:
            # remove bbox regression weights/bias for bg class
            old_weight = model[k]
            new_weight = old_weight[4:]
            logger.warning(
                "Change {} from shape {} to {}.".format(
                    k, tuple(old_weight.shape), tuple(new_weight.shape)
                )
            )
        elif "roi_heads.mask_head.predictor." in k:
            # remove mask prediction weights for bg class
            old_weight = model[k]
            new_weight = old_weight[1:]
            logger.warning(
                "Change {} from shape {} to {}.".format(
                    k, tuple(old_weight.shape), tuple(new_weight.shape)
                )
            )
        elif "roi_heads.box_predictor.cls_score." in k:
            # move classification weights for bg class from the first index to the last index
            old_weight = model[k]
            new_weight = np.concatenate((old_weight[1:], old_weight[:1]))
            logger.warning(
                "Change BG in {} from index 0 to index {}.".format(k, new_weight.shape[0] - 1)
            )
        else:
            continue
        model[k] = new_weight
    return model


def _convert_rpn_name(model):
    """
    Automatically convert the model for breakage in D15084700.
    """
    # TODO This is temporary. Remove this function after a while

    keys = model.keys()
    rpn_keys = [k for k in keys if "rpn.head." in k or "rpn.anchor_generator" in k]

    if len(rpn_keys) == 0:
        return model

    logger = logging.getLogger(__name__)
    logger.warning(
        "Your rpn weight names are in an old format! Please see D15084700 "
        "and convert your weight names!"
    )
    logger.warning("Now attempting to automatically convert the weight names for you ...")

    for old_key in rpn_keys:
        new_key = old_key.replace("rpn.head.", "proposal_generator.rpn_head.").replace(
            "rpn.anchor_", "proposal_generator.anchor_"
        )
        logger.warning("Rename parameter '{}' in checkpoint to '{}'.".format(old_key, new_key))
        model[new_key] = model.pop(old_key)

    return model


class DetectionCheckpointer(Checkpointer):
    """
    A checkpointer that is able to handle models in detectron & detectron2
    model zoo, and apply conversions for legacy models.
    """

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                data["model"] = _convert_rpn_name(data["model"])
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        loaded["model"] = _convert_background_class(loaded["model"])
        loaded["model"] = _convert_rpn_name(loaded["model"])
        return loaded

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            model_state_dict = self.model.state_dict()
            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict
        # for non-caffe2 models, use standard ways to load it
        super()._load_model(checkpoint)

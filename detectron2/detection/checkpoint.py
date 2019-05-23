import copy
import logging
import numpy as np
import pickle
import re

from detectron2.utils.c2_model_loading import convert_basic_c2_names
from detectron2.utils.checkpoint import Checkpointer


# TODO make it support RetinaNet, etc
def _convert_c2_detectron_names(weights):
    """
    Map Caffe2 Detectron weight names to Detectron2 names.

    Args:
        weights (dict): name -> numpy array

    Returns:
        dict with detectron2 names
    """
    logger = logging.getLogger(__name__)
    logger.info("Remapping C2 weights ......")
    original_keys = sorted(weights.keys())
    layer_keys = copy.deepcopy(original_keys)

    layer_keys = convert_basic_c2_names(layer_keys)

    # --------------------------------------------------------------------------
    # RPN hidden representation conv
    # --------------------------------------------------------------------------
    # FPN case
    # In the C2 model, the RPN hidden layer conv is defined for FPN level 2 and then
    # shared for all other levels, hence the appearance of "fpn2"
    layer_keys = [
        k.replace("conv.rpn.fpn2", "proposal_generator.rpn_head.conv") for k in layer_keys
    ]
    # Non-FPN case
    layer_keys = [k.replace("conv.rpn", "proposal_generator.rpn_head.conv") for k in layer_keys]

    # --------------------------------------------------------------------------
    # RPN box transformation conv
    # --------------------------------------------------------------------------
    # FPN case (see note above about "fpn2")
    layer_keys = [
        k.replace("rpn.bbox.pred.fpn2", "proposal_generator.rpn_head.anchor_deltas")
        for k in layer_keys
    ]
    layer_keys = [
        k.replace("rpn.cls.logits.fpn2", "proposal_generator.rpn_head.objectness_logits")
        for k in layer_keys
    ]
    # Non-FPN case
    layer_keys = [
        k.replace("rpn.bbox.pred", "proposal_generator.rpn_head.anchor_deltas") for k in layer_keys
    ]
    layer_keys = [
        k.replace("rpn.cls.logits", "proposal_generator.rpn_head.objectness_logits")
        for k in layer_keys
    ]

    # --------------------------------------------------------------------------
    # Fast R-CNN box head
    # --------------------------------------------------------------------------
    layer_keys = [re.sub("^bbox\\.pred", "bbox_pred", k) for k in layer_keys]
    layer_keys = [re.sub("^cls\\.score", "cls_score", k) for k in layer_keys]
    layer_keys = [re.sub("^fc6\\.", "box_head.fc1.", k) for k in layer_keys]
    layer_keys = [re.sub("^fc7\\.", "box_head.fc2.", k) for k in layer_keys]
    # 4conv1fc head tensor names: head_conv1_w, head_conv1_gn_s
    layer_keys = [re.sub("^head\\.conv", "box_head.conv", k) for k in layer_keys]

    # --------------------------------------------------------------------------
    # FPN lateral and output convolutions
    # --------------------------------------------------------------------------
    def fpn_map(name):
        """
        Look for keys with the following patterns:
        1) Starts with "fpn.inner."
           Example: "fpn.inner.res2.2.sum.lateral.weight"
           Meaning: These are lateral pathway convolutions
        2) Starts with "fpn.res"
           Example: "fpn.res2.2.sum.weight"
           Meaning: These are FPN output convolutions
        """
        splits = name.split(".")
        norm = ".norm" if "norm" in splits else ""
        if name.startswith("fpn.inner."):
            # splits example: ['fpn', 'inner', 'res2', '2', 'sum', 'lateral', 'weight']
            stage = int(splits[2][len("res") :])
            return "fpn_lateral{}{}.{}".format(stage, norm, splits[-1])
        elif name.startswith("fpn.res"):
            # splits example: ['fpn', 'res2', '2', 'sum', 'weight']
            stage = int(splits[1][len("res") :])
            return "fpn_output{}{}.{}".format(stage, norm, splits[-1])
        return name

    layer_keys = [fpn_map(k) for k in layer_keys]

    # --------------------------------------------------------------------------
    # Mask R-CNN mask head
    # --------------------------------------------------------------------------
    # roi_heads.StandardROIHeads case
    layer_keys = [k.replace(".[mask].fcn", "mask_head.mask_fcn") for k in layer_keys]
    layer_keys = [re.sub("^\\.mask\\.fcn", "mask_head.mask_fcn", k) for k in layer_keys]
    layer_keys = [k.replace("mask.fcn.logits", "mask_head.predictor") for k in layer_keys]
    # roi_heads.Res5ROIHeads case
    layer_keys = [k.replace("conv5.mask", "mask_head.deconv") for k in layer_keys]

    # --------------------------------------------------------------------------
    # Keypoint R-CNN head
    # --------------------------------------------------------------------------
    # interestingly, the keypoint head convs have blob names that are simply "conv_fcnX"
    layer_keys = [k.replace("conv.fcn", "roi_heads.keypoint_head.conv_fcn") for k in layer_keys]
    layer_keys = [
        k.replace("kps.score.lowres", "roi_heads.keypoint_head.score_lowres") for k in layer_keys
    ]
    layer_keys = [k.replace("kps.score.", "roi_heads.keypoint_head.score.") for k in layer_keys]

    # --------------------------------------------------------------------------
    # Done with replacements
    # --------------------------------------------------------------------------
    assert len(set(layer_keys)) == len(layer_keys)

    max_c2_key_size = max(len(k) for k in original_keys)
    new_weights = {}
    for orig, renamed in zip(original_keys, layer_keys):
        logger.info("C2 name: {: <{}} mapped name: {}".format(orig, max_c2_key_size, renamed))
        if renamed.startswith("bbox_pred.") or renamed.startswith("mask_head.predictor."):
            # remove the meaningless prediction weight for background class
            new_start_idx = 4 if renamed.startswith("bbox_pred.") else 1
            new_weights[renamed] = weights[orig][new_start_idx:]
            logger.info(
                "Remove prediction weight for background class in {}. The shape changes from "
                "{} to {}.".format(
                    renamed, tuple(weights[orig].shape), tuple(new_weights[renamed].shape)
                )
            )
        elif renamed.startswith("cls_score."):
            # move weights of bg class from original index 0 to last index
            logger.info(
                "Move classification weights for background class in {} from index 0 to "
                "index {}.".format(renamed, weights[orig].shape[0] - 1)
            )
            new_weights[renamed] = np.concatenate((weights[orig][1:], weights[orig][:1]))
        else:
            new_weights[renamed] = weights[orig]

    return new_weights


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
        logger.warning("Change origin rpn weight name {} to {}.".format(old_key, new_key))
        model[new_key] = model.pop(old_key)

    return model


class DetectionCheckpointer(Checkpointer):
    def _load_file(self, f):
        if f.endswith(".pkl"):
            data = pickle.load(open(f, "rb"), encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data.pop("__author__")))
                data["model"] = _convert_rpn_name(data["model"])
                return data
            else:
                # assume file is from Caffe2
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                model = _convert_c2_detectron_names(data)
            return {"model": model}
        loaded = super()._load_file(f)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        loaded["model"] = _convert_background_class(loaded["model"])
        loaded["model"] = _convert_rpn_name(loaded["model"])
        return loaded

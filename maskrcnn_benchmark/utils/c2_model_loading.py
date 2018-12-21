import logging
import pickle
import copy
import re

import torch


def _load_c2_pickled_weights(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    if "blobs" in data:
        weights = data["blobs"]
    else:
        weights = data
    return weights


def _convert_basic_c2_names(original_keys):
    """
    Apply some basic name conversion to C2 weights.

    Args:
        original_keys (list[str]):
    Returns:
        list[str]: The same number of strings matching those in original_keys.
    """
    layer_keys = copy.deepcopy(original_keys)
    layer_keys = [{
        'pred_b': 'fc1000_b',
        'pred_w': 'fc1000_w'
    }.get(k, k) for k in layer_keys]   # some hard-coded mappings

    layer_keys = [k.replace("_", ".") for k in layer_keys]
    layer_keys = [re.sub('.b$', '.bias', k) for k in layer_keys]
    layer_keys = [re.sub('.w$', '.weight', k) for k in layer_keys]
    layer_keys = [re.sub('bn.s$', 'bn.weight', k) for k in layer_keys]

    # stages
    layer_keys = [re.sub("^res.conv1.bn.", "bn1.", k) for k in layer_keys]
    layer_keys = [re.sub("^res2.", "layer1.", k) for k in layer_keys]
    layer_keys = [re.sub("^res3.", "layer2.", k) for k in layer_keys]
    layer_keys = [re.sub("^res4.", "layer3.", k) for k in layer_keys]
    layer_keys = [re.sub("^res5.", "layer4.", k) for k in layer_keys]

    # blocks
    layer_keys = [k.replace(".branch1.bn.", ".downsample.1.") for k in layer_keys]
    layer_keys = [k.replace(".branch1.", ".downsample.0.") for k in layer_keys]
    layer_keys = [k.replace(".branch2a.bn.", ".bn1.") for k in layer_keys]
    layer_keys = [k.replace(".branch2a.", ".conv1.") for k in layer_keys]
    layer_keys = [k.replace(".branch2b.bn.", ".bn2.") for k in layer_keys]
    layer_keys = [k.replace(".branch2b.", ".conv2.") for k in layer_keys]
    layer_keys = [k.replace(".branch2c.bn.", ".bn3.") for k in layer_keys]
    layer_keys = [k.replace(".branch2c.", ".conv3.") for k in layer_keys]
    return layer_keys


def _convert_c2_detectron_weights(weights):
    """
    Apply conversion to C2 Detectrno weights, after applying the basic conversion.

    Args:
        weights (dict): name->numpy array
    Returns:
        dict
    """
    logger = logging.getLogger(__name__)
    logger.info("Remapping C2 weights ......")
    original_keys = sorted(weights.keys())
    layer_keys = copy.deepcopy(original_keys)

    layer_keys = _convert_basic_c2_names(layer_keys)

    # RPN / Faster RCNN
    layer_keys = [k.replace("conv.rpn.fpn2", "rpn.head.conv") for k in layer_keys]
    layer_keys = [k.replace("conv.rpn", "rpn.head.conv") for k in layer_keys]

    layer_keys = [k.replace("rpn.bbox.pred.fpn2", "rpn.head.bbox_pred") for k in layer_keys]
    layer_keys = [k.replace("rpn.cls.logits.fpn2", "rpn.head.cls_logits") for k in layer_keys]
    layer_keys = [k.replace("rpn.bbox.pred", "rpn.head.bbox_pred") for k in layer_keys]
    layer_keys = [k.replace("rpn.cls.logits", "rpn.head.cls_logits") for k in layer_keys]

    # Fast R-CNN
    layer_keys = [re.sub("^bbox.pred", "bbox_pred", k) for k in layer_keys]
    layer_keys = [re.sub("^cls.score", "cls_score", k) for k in layer_keys]
    layer_keys = [re.sub("^fc6.", "box_head.fc1.", k) for k in layer_keys]
    layer_keys = [re.sub("^fc7.", "box_head.fc2.", k) for k in layer_keys]

    # FPN
    def fpn_map(name):
        splits = name.split('.')
        if name.startswith('fpn.inner.'):
            # fpn_inner_res2_2_sum_lateral_b
            stage = int(splits[2][-1]) - 1
            return "fpn_inner{}.{}".format(stage, splits[-1])
        elif name.startswith('fpn.res'):
            # fpn_res2_2_sum_b
            stage = int(splits[1][-1]) - 1
            return "fpn_layer{}.{}".format(stage, splits[-1])
        return name

    layer_keys = [fpn_map(k) for k in layer_keys]

    # Mask R-CNN
    layer_keys = [k.replace(".[mask].fcn", "mask_head.mask_fcn") for k in layer_keys]
    layer_keys = [k.replace("mask.fcn.logits", "mask_head.predictor") for k in layer_keys]
    layer_keys = [k.replace("conv5.mask", "mask_head.deconv") for k in layer_keys]
    assert len(set(layer_keys)) == len(layer_keys)

    max_c2_key_size = max([len(k) for k in original_keys])
    new_weights = {}
    for orig, renamed in zip(original_keys, layer_keys):
        logger.info("C2 name: {: <{}} mapped name: {}".format(orig, max_c2_key_size, renamed))
        new_weights[renamed] = weights[orig]
    return new_weights


def load_c2_format(filename):
    """
    Load a caffe2 checkpoint.

    Args:
        filename (str):
    Returns:
        dict: name -> torch.Tensor
    """
    # TODO make it support other architectures
    state_dict = _load_c2_pickled_weights(filename)
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith('_momentum')}
    state_dict = _convert_c2_detectron_weights(state_dict)
    state_dict = {k: torch.from_numpy(v) for k, v in state_dict.items()}
    return dict(model=state_dict)

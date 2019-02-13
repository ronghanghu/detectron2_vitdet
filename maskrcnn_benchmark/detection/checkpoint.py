import copy
import logging
import re

from maskrcnn_benchmark.utils.c2_model_loading import convert_basic_c2_names, load_c2_weights
from maskrcnn_benchmark.utils.checkpoint import Checkpointer

from .config import paths_catalog


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
    layer_keys = [k.replace("conv.rpn.fpn2", "rpn.head.conv") for k in layer_keys]
    # Non-FPN case
    layer_keys = [k.replace("conv.rpn", "rpn.head.conv") for k in layer_keys]

    # --------------------------------------------------------------------------
    # RPN box transformation conv
    # --------------------------------------------------------------------------
    # FPN case (see note above about "fpn2")
    layer_keys = [k.replace("rpn.bbox.pred.fpn2", "rpn.head.anchor_deltas") for k in layer_keys]
    layer_keys = [
        k.replace("rpn.cls.logits.fpn2", "rpn.head.objectness_logits") for k in layer_keys
    ]
    # Non-FPN case
    layer_keys = [k.replace("rpn.bbox.pred", "rpn.head.anchor_deltas") for k in layer_keys]
    layer_keys = [k.replace("rpn.cls.logits", "rpn.head.objectness_logits") for k in layer_keys]

    # --------------------------------------------------------------------------
    # Fast R-CNN box head
    # --------------------------------------------------------------------------
    layer_keys = [re.sub("^bbox.pred", "bbox_pred", k) for k in layer_keys]
    layer_keys = [re.sub("^cls.score", "cls_score", k) for k in layer_keys]
    layer_keys = [re.sub("^fc6.", "box_head.fc1.", k) for k in layer_keys]
    layer_keys = [re.sub("^fc7.", "box_head.fc2.", k) for k in layer_keys]

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
        if name.startswith("fpn.inner."):
            # splits example: ['fpn', 'inner', 'res2', '2', 'sum', 'lateral', 'weight']
            stage = int(splits[2][len("res") :])
            return "fpn_lateral{}.{}".format(stage, splits[-1])
        elif name.startswith("fpn.res"):
            # splits example: ['fpn', 'res2', '2', 'sum', 'weight']
            stage = int(splits[1][len("res") :])
            return "fpn_output{}.{}".format(stage, splits[-1])
        return name

    layer_keys = [fpn_map(k) for k in layer_keys]

    # --------------------------------------------------------------------------
    # Mask R-CNN mask head
    # --------------------------------------------------------------------------
    # roi_heads.StandardROIHeads case
    layer_keys = [k.replace(".[mask].fcn", "mask_head.mask_fcn") for k in layer_keys]
    layer_keys = [k.replace("mask.fcn.logits", "mask_head.predictor") for k in layer_keys]
    # roi_heads.Res5ROIHeads case
    layer_keys = [k.replace("conv5.mask", "mask_head.deconv") for k in layer_keys]

    # --------------------------------------------------------------------------
    # Done with replacements
    # --------------------------------------------------------------------------
    assert len(set(layer_keys)) == len(layer_keys)

    max_c2_key_size = max(len(k) for k in original_keys)
    new_weights = {}
    for orig, renamed in zip(original_keys, layer_keys):
        logger.info("C2 name: {: <{}} mapped name: {}".format(orig, max_c2_key_size, renamed))
        new_weights[renamed] = weights[orig]
    return new_weights


class DetectionCheckpointer(Checkpointer):
    def __init__(self, model, optimizer=None, scheduler=None, save_dir="", save_to_disk=None):
        super().__init__(model, optimizer, scheduler, save_dir, save_to_disk)

    # TODO catalog lookup may be moved to base Checkpointer
    def _download_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        return super()._download_file(f)

    def _load_file(self, f):
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            model = load_c2_weights(f)
            model = _convert_c2_detectron_names(model)
            return {"model": model}
        # load native detectron.pytorch checkpoint
        loaded = super()._load_file(f)
        if "model" not in loaded:
            loaded = {"model": model}
        return loaded

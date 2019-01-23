import copy
import logging
import re

from maskrcnn_benchmark.utils.c2_model_loading import convert_basic_c2_names, load_c2_weights
from maskrcnn_benchmark.utils.checkpoint import Checkpointer
from maskrcnn_benchmark.utils.imports import import_file


# TODO make it support RetinaNet, etc
def _convert_c2_detectron_weights(weights):
    """
    Apply conversion to C2 Detectron weights.

    Args:
        weights (dict): name->numpy array
    Returns:
        dict
    """
    logger = logging.getLogger(__name__)
    logger.info("Remapping C2 weights ......")
    original_keys = sorted(weights.keys())
    layer_keys = copy.deepcopy(original_keys)

    layer_keys = convert_basic_c2_names(layer_keys)

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
        splits = name.split(".")
        if name.startswith("fpn.inner."):
            # fpn_inner_res2_2_sum_lateral_b
            stage = int(splits[2][-1])
            return "fpn_lateral{}.{}".format(stage, splits[-1])
        elif name.startswith("fpn.res"):
            # fpn_res2_2_sum_b
            stage = int(splits[1][-1])
            return "fpn_output{}.{}".format(stage, splits[-1])
        return name

    layer_keys = [fpn_map(k) for k in layer_keys]

    # Mask R-CNN
    layer_keys = [k.replace(".[mask].fcn", "mask_head.mask_fcn") for k in layer_keys]
    layer_keys = [k.replace("mask.fcn.logits", "mask_head.predictor") for k in layer_keys]
    layer_keys = [k.replace("conv5.mask", "mask_head.deconv") for k in layer_keys]
    assert len(set(layer_keys)) == len(layer_keys)

    max_c2_key_size = max(len(k) for k in original_keys)
    new_weights = {}
    for orig, renamed in zip(original_keys, layer_keys):
        logger.info("C2 name: {: <{}} mapped name: {}".format(orig, max_c2_key_size, renamed))
        new_weights[renamed] = weights[orig]
    return new_weights


class DetectionCheckpointer(Checkpointer):
    def __init__(self, cfg, model, optimizer=None, scheduler=None, save_dir="", save_to_disk=None):
        super().__init__(model, optimizer, scheduler, save_dir, save_to_disk)
        self.cfg = cfg.clone()

    # TODO catalog lookup may be moved to base Checkpointer
    def _download_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        return super()._download_file(f)

    def _load_file(self, f):
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            model = load_c2_weights(f)
            model = _convert_c2_detectron_weights(model)
            return dict(model=model)
        # load native detectron.pytorch checkpoint
        loaded = super()._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded

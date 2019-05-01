import copy
import logging
import numpy as np
import re

from detectron2.utils.c2_model_loading import convert_basic_c2_names, load_c2_weights
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


class DetectionCheckpointer(Checkpointer):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        cache_on_load=False,
    ):
        super().__init__(model, optimizer, scheduler, save_dir, save_to_disk, cache_on_load)

    def _resolve_path(self, f):
        # Resolve "catalog://" paths
        if f.startswith("catalog://"):
            catalog_f = ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        return f

    def _load_file(self, f):
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            model = load_c2_weights(f)
            model = _convert_c2_detectron_names(model)
            return {"model": model}
        # load native detectron.pytorch checkpoint
        loaded = super()._load_file(f)
        if "model" not in loaded:
            loaded = {"model": loaded}
        loaded["model"] = _convert_background_class(loaded["model"])
        return loaded


class ModelCatalog(object):
    S3_C2_DETECTRON_PREFIX = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "FAIR/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_PATH_FORMAT = (
        "{prefix}/{url}/output/train/{dataset}/{type}/model_final.pkl"
    )  # noqa B950

    C2_DATASET_COCO = "coco_2014_train%3Acoco_2014_valminusminival"
    C2_DATASET_COCO_KEYPOINTS = "keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival"

    # format: {model_name} -> part of the url
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "35857197/12_2017_baselines/e2e_faster_rcnn_R-50-C4_1x.yaml.01_33_49.iAX0mXvW",  # noqa B950
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "35857345/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml.01_36_30.cUF7QR7I",  # noqa B950
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "35857890/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_1x.yaml.01_38_50.sNxI7sX7",  # noqa B950
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "36761737/12_2017_baselines/e2e_faster_rcnn_X-101-32x8d-FPN_1x.yaml.06_31_39.5MIHi1fZ",  # noqa B950
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "35858791/12_2017_baselines/e2e_mask_rcnn_R-50-C4_1x.yaml.01_45_57.ZgkA7hPB",  # noqa B950
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "35858933/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC",  # noqa B950
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "35861795/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml.02_31_37.KqyEK4tT",  # noqa B950
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "36761843/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_1x.yaml.06_35_59.RZotkLKI",  # noqa B950
        "48616381/e2e_mask_rcnn_R-50-FPN_2x_gn": "GN/48616381/04_2018_gn_baselines/e2e_mask_rcnn_R-50-FPN_2x_gn_0416.13_23_38.bTlTI97Q",  # noqa B950
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "37697547/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_1x.yaml.08_42_54.kdzV35ao",  # noqa B950
        "35998355/rpn_R-50-C4_1x": "35998355/12_2017_baselines/rpn_R-50-C4_1x.yaml.08_00_43.njH5oD9L",  # noqa B950
        "35998814/rpn_R-50-FPN_1x": "35998814/12_2017_baselines/rpn_R-50-FPN_1x.yaml.08_06_03.Axg0r179",  # noqa B950
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog._get_c2_detectron_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog._get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def _get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_PREFIX
        name = name[len("ImageNetPretrained/") :]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def _get_c2_detectron_baselines(name):
        name = name[len("Caffe2Detectron/COCO/") :]
        url = ModelCatalog.C2_DETECTRON_MODELS[name]
        if "keypoint_rcnn" in name:
            dataset = ModelCatalog.C2_DATASET_COCO_KEYPOINTS
        else:
            dataset = ModelCatalog.C2_DATASET_COCO

        if "35998355/rpn_R-50-C4_1x" in name:
            # this one model is somehow different from others ..
            type = "rpn"
        else:
            type = "generalized_rcnn"

        # Detectron C2 models are stored in the structure defined in `C2_DETECTRON_PATH_FORMAT`.
        url = ModelCatalog.C2_DETECTRON_PATH_FORMAT.format(
            prefix=ModelCatalog.S3_C2_DETECTRON_PREFIX, url=url, type=type, dataset=dataset
        )
        return url

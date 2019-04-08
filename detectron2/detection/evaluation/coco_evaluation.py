import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.inference import DatasetEvaluator


class COCOEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection/segmentation, keypoint detection
    outputs using COCO's metrics and APIs.
    """

    def __init__(self, dataset_split, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_split (str): name of a dataset split to be evaluated.
                It must has the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                    "dataset_name": the name of the dataset it belongs to.
            cfg (Config): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        split_meta = MetadataCatalog.get(dataset_split)
        self._coco_api = COCO(split_meta.json_file)
        self._dataset_meta = MetadataCatalog.get(split_meta.dataset_name)

        self.kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS

    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        if cfg.MODEL.RPN_ONLY:
            tasks = ("box_proposals",)
        else:
            tasks = ("bbox",)
            if cfg.MODEL.MASK_ON:
                tasks = tasks + ("segm",)
            if cfg.MODEL.KEYPOINT_ON:
                tasks = tasks + ("keypoints",)
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is either list of :class:`Instances` or
                list of dicts with key "detector" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            if isinstance(output, dict):
                output = output["detector"]
            output = output.to(self._cpu_device)
            if output.has("pred_masks"):
                # use RLE to encode the masks, because they are too large and takes memory
                # since this evaluator stores outputs of the entire dataset
                rles = [
                    mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
                    for mask in output.pred_masks
                ]
                for rle in rles:
                    # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                    # json writer which always produces strings cannot serialize a bytestream
                    # unless you decode it. Thankfully, utf-8 works out (which is also what
                    # the pycocotools/_mask.pyx does).
                    rle["counts"] = rle["counts"].decode("utf-8")
                output.pred_masks_rle = rles
                output.remove("pred_masks")

            prediction = {"image_id": input["image_id"], "instances": output}
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            synchronize()
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))

            if not is_main_process():
                return

        if self._output_dir:
            torch.save(self._predictions, os.path.join(self._output_dir, "predictions.pth"))

        tasks = set(self._tasks)
        self._results = OrderedDict()
        if "box_proposals" in tasks:
            self._eval_box_proposals()
            tasks.remove("box_proposals")

        if len(tasks) != 0:
            self._eval_predictions(tasks)

        self._logger.info(self._results)
        return self._results

    def _eval_predictions(self, tasks):
        """
        Evaluate self._predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format")
        coco_results = prepare_for_coco_evaluation(self._predictions)

        # unmap the category ids for COCO
        if hasattr(self._dataset_meta, "json_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._dataset_meta.json_id_to_contiguous_id.items()
            }
            for result in coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_results.json")
            f = open(file_path, "w")
        else:
            f = tempfile.NamedTemporaryFile(suffix=".json")
            file_path = f.name
        with f:
            json.dump(coco_results, f)
            f.flush()

            self._logger.info("Evaluating predictions")
            for task in sorted(tasks):
                res = evaluate_predictions_on_coco(
                    self._coco_api, coco_results, file_path, self.kpt_oks_sigmas, task
                )
                self._results[task] = res

    def _eval_box_proposals(self):
        """
        Evaluate the box proposals in self._predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        self._logger.info("Evaluating bbox proposals")
        res = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    self._predictions, self._coco_api, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = stats["ar"].item() * 100
        self._results["box_proposals"] = res


def prepare_for_coco_evaluation(dataset_predictions):
    """
    Args:
        dataset_predictions (list[dict]): the same format as returned by `inference_on_dataset`.

    Returns:
        list[dict]: the format used by COCO evaluation.
    """

    coco_results = []
    processed_ids = {}
    for prediction_dict in dataset_predictions:
        img_id = prediction_dict["image_id"]
        if img_id in processed_ids:
            # The same image may be processed multiple times due to the underlying
            # dataset sampler, such as torch.utils.data.distributed.DistributedSampler:
            # https://git.io/fhScl
            continue
        processed_ids[img_id] = True
        predictions = prediction_dict["instances"]
        num_instance = len(predictions)
        if num_instance == 0:
            continue

        boxes = predictions.pred_boxes.tensor.numpy()
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = predictions.scores.tolist()
        classes = predictions.pred_classes.tolist()

        has_mask = predictions.has("pred_masks_rle")
        if has_mask:
            rles = predictions.pred_masks_rle

        has_keypoints = predictions.has("pred_keypoints")
        if has_keypoints:
            keypoints = predictions.pred_keypoints.tensor.flatten(1).tolist()

        for k in range(num_instance):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
            }
            if has_mask:
                result["segmentation"] = rles[k]
            if has_keypoints:
                result["keypoints"] = keypoints[k]
            coco_results.append(result)
    return coco_results


# inspired from Detectron
def evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None, area="all", limit=None):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["instances"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = predictions.objectness_logits.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
        anno = coco_api.loadAnns(ann_ids)
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno
            if obj["iscrowd"] == 0
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(predictions) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def evaluate_predictions_on_coco(
    coco_gt, coco_results, json_result_file, kpt_oks_sigmas, iou_type="bbox"
):
    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }[iou_type]

    if len(coco_results) == 0:  # cocoapi does not handle empty results very well
        return {metric: -1 for metric in metrics}

    coco_dt = coco_gt.loadRes(str(json_result_file))
    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {metric: coco_eval.stats[idx] * 100 for idx, metric in enumerate(metrics)}

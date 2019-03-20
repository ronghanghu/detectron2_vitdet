import datetime
import itertools
import json
import logging
import numpy as np
import os
import tempfile
import time
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from detectron2.data import DatasetFromList
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.comm import all_gather, is_main_process, synchronize

from .data import DatasetCatalog, build_detection_test_loader


def compute_on_dataset(model, data_loader, aggregate_across_ranks=True):
    """
    Run model on the dataloader and returns all outputs.

    Args:
        model: a callable that takes an object from data_loader and returns some outputs
        data_loader: an iterable with a length
        aggregate_across_ranks (bool): if True, the results will be
            concatenated across all ranks. This is useful when all ranks each
            run on a disjoint subset of a large dataset.
            If False, each rank will return only the results it computed on the given data_loader.

    Returns:
        list[dict]. Each dict has: "image_id", "original_height", "original_width", "instances".
            "instances" is a `Boxes` containing the post-processed outputs of the model.
    """
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start evaluation on {} images".format(len(data_loader)))
    start_time = time.time()

    old_training_mode = model.training
    model.eval()

    dataset_predictions = []
    cpu_device = torch.device("cpu")
    for batch in tqdm(data_loader):
        _, __, dataset_dicts = batch
        with torch.no_grad():
            outputs = model(batch)
            for dataset_dict, output in zip(dataset_dicts, outputs):
                output = output.to(cpu_device)

                if output.has("pred_masks"):
                    # use RLE to encode the masks, because they are too large and takes memory when
                    # running over an entire dataset
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

                prediction_dict = {k: dataset_dict[k] for k in ["height", "width", "image_id"]}
                prediction_dict["instances"] = output
                dataset_predictions.append(prediction_dict)

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time / len(data_loader), num_devices
        )
    )

    synchronize()

    if aggregate_across_ranks:
        dataset_predictions = all_gather(dataset_predictions)
        dataset_predictions = list(itertools.chain(*dataset_predictions))
    model.train(old_training_mode)

    return dataset_predictions


def prepare_for_coco_evaluation(dataset_predictions):
    """
    Args:
        dataset_predictions (list[dict]): the same format as returned by `compute_on_dataset`.

    Returns:
        list[dict]: the format used by COCO evaluation.
    """

    coco_results = []
    processed_ids = {}
    for prediction_dict in tqdm(dataset_predictions):
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
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes, mode="xywh").clone(mode="xyxy")
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


def evaluate_predictions_on_coco(coco_gt, coco_results, json_result_file, iou_type="bbox"):
    coco_dt = coco_gt.loadRes(str(json_result_file))
    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx] * 100  # x100 makes numbers more readable

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def coco_evaluation(cfg, model, dataset_name, output_folder=None):
    """
    This function returns nothing on non-master processes (but still needs to be called).
    On master process, it returns the following:

    Args:
        model: the detection model
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog

    Returns:
       dict of dict: results[task][metric] is a float.
       list[dict]: one for each image, contains the outputs from the model.
       list[dict]: one for each instance, in the COCO evaluation format.
    """
    coco_dataset = DatasetFromList(DatasetCatalog.get(dataset_name))

    data_loader = build_detection_test_loader(cfg, coco_dataset)
    dataset_predictions = compute_on_dataset(model, data_loader, aggregate_across_ranks=True)
    if not is_main_process():
        return

    if output_folder:
        torch.save(dataset_predictions, os.path.join(output_folder, "predictions.pth"))

    logger = logging.getLogger(__name__)

    from pycocotools.coco import COCO

    json_file = DatasetCatalog.get_coco_path(dataset_name)["json_file"]
    coco_api = COCO(json_file)

    if cfg.MODEL.RPN_ONLY:
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        results = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    dataset_predictions, coco_api, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                results.results["box_proposal"][key] = stats["ar"].item()
        logger.info(results)
        if output_folder:
            torch.save(results, os.path.join(output_folder, "box_proposals.pth"))
        recalls = results.results["box_proposal"]
        for key in recalls:
            recalls[key] = recalls[key] * 100
        return (results.results,)

    logger.info("Preparing results for COCO format")
    coco_results = prepare_for_coco_evaluation(dataset_predictions)

    # unmap the category ids for COCO
    dataset_meta = DatasetCatalog.get_metadata(dataset_name)
    if hasattr(dataset_meta, "json_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in dataset_meta.json_id_to_contiguous_id.items()}
        for result in coco_results:
            result["category_id"] = reverse_id_mapping[result["category_id"]]

    with tempfile.NamedTemporaryFile() as f:
        file_path = f.name
        if output_folder:
            file_path = os.path.join(output_folder, "coco_results.json")
        with open(file_path, "w") as f:
            json.dump(coco_results, f)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)

    results = COCOResults(*iou_types)
    if len(coco_results) == 0:
        return results.results, coco_results, dataset_predictions

    logger.info("Evaluating predictions")

    for iou_type in iou_types:
        res = evaluate_predictions_on_coco(coco_api, coco_results, file_path, iou_type)
        results.update(res)
    results = results.results  # get the OrderedDict from COCOResults
    logger.info(results)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))

    return results, coco_results, dataset_predictions


def print_copypaste_format(results):
    """
    Print results in a format that's easy to copypaste to excel.

    Args:
        results: OrderedDict
    """
    assert isinstance(results, OrderedDict)  # unordered results cannot be properly printed
    logger = logging.getLogger(__name__)
    for task in ["bbox", "segm", "keypoints", "box_proposal"]:
        if task not in results:
            continue
        res = results[task]
        logger.info("copypaste: Task: {}".format(task))
        logger.info("copypaste: " + ",".join([n for n in res.keys()]))
        logger.info("copypaste: " + ",".join(["{0:.4f}".format(v) for v in res.values()]))
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
from PIL import Image
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets import COCOMeta
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.comm import is_main_process, scatter_gather, synchronize

from .modeling.roi_heads.paste_mask import Masker  # TODO move inside models


def postprocess(result, original_width, original_height):
    result = result.copy_with_fields(result.fields())
    MASK_THRESHOLD = 0.5
    # we assume we did the scaling without changing the aspect ratio
    scale = np.sqrt(result.size[0] * 1.0 / original_width * result.size[1] / original_height)
    result.size = (original_width, original_height)
    bbox = result.bbox = result.bbox / scale
    result.clip_to_image()
    # maybe clip bbox again

    if result.has_field("mask"):
        masks = result.get_field("mask")  # N, 1, M, M
        if True:
            masker = Masker(threshold=MASK_THRESHOLD, padding=1)
            pasted_masks = [x[0] for x in masker.forward_single_image(masks, result)]
        else:
            pasted_masks = []
            for box, mask in zip(bbox, masks):
                # This quantization logic is copied from tensorpack/examples/FasterRCNN

                # First, translate floating point coord to integer coord
                # box fpcoor=0.0 -> intcoor=0.0
                x0, y0 = list(map(int, box[:2] + 0.5))
                # box fpcoor=h -> intcoor=h-1, inclusive
                x1, y1 = list(map(int, box[2:] - 0.5))  # inclusive
                x1 = max(x0, x1)  # require at least 1x1
                y1 = max(y0, y1)
                w = x1 + 1 - x0
                h = y1 + 1 - y0

                # rounding errors could happen here, because masks were not originally computed for this shape.
                # but it's hard to do better, because the network does not know the "original" scale
                mask = Image.fromarray(mask.numpy()[0]).resize((w, h))
                ret = np.zeros((original_height, original_width), dtype="uint8")
                ret[y0 : y1 + 1, x0 : x1 + 1] = (np.asarray(mask) > MASK_THRESHOLD).astype("uint8")
                pasted_masks.append(ret)
        result.add_field("mask", pasted_masks)
    return result


def compute_on_dataset(model, data_loader):
    """
    Returns:
        list[dict]. Each dict has: "id", "original_height", "original_width", "result".
            "result" is a `BoxList` containing the outputs of the model.
    """
    model.eval()
    results = []
    cpu_device = torch.device("cpu")
    for batch in tqdm(data_loader):
        _, __, roidbs = batch
        with torch.no_grad():
            outputs = model(batch)
            for roidb, output in zip(roidbs, outputs):
                output = output.to(cpu_device)
                result = {k: roidb[k] for k in ["original_height", "original_width", "id"]}
                result["result"] = output
                results.append(result)
    return results


def prepare_for_coco_evaluation(predictions):
    """
    Args:
        predictions (list[dict]): the same format as returned by `compute_on_dataset`.

    Returns:
        list[dict]: the format used by COCO evaluation.
    """

    coco_results = []
    for roidb in tqdm(predictions):
        prediction = roidb["result"]
        num_instance = len(prediction)
        if num_instance == 0:
            continue

        prediction = postprocess(prediction, roidb["original_width"], roidb["original_height"])
        prediction = prediction.convert("xywh")
        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        has_mask = prediction.has_field("mask")
        if has_mask:
            masks = prediction.get_field("mask")

            rles = [
                mask_util.encode(np.array(mask[:, :, np.newaxis], order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [COCOMeta().contiguous_id_to_json_id[i] for i in labels]

        for k in range(num_instance):
            result = {
                "image_id": roidb["id"],
                "category_id": mapped_labels[k],
                "bbox": boxes[k],
                "score": scores[k],
            }
            if has_mask:
                result["segmentation"] = rles[k]
            coco_results.append(result)
    return coco_results


# inspired from Detectron
# TODO(yuxin) this has not been tested for a long time; should not use "dataset"
def evaluate_box_proposals(predictions, dataset, thresholds=None, area="all", limit=None):
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

    for roidb in predictions:
        prediction = roidb["result"]
        image_width = roidb["original_width"]
        image_height = roidb["original_height"]
        prediction = postprocess(prediction, roidb["original_width"], roidb["original_height"])

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("objectness").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.ds.coco.getAnnIds(imgIds=roidb["id"])
        anno = dataset.ds.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert("xyxy")
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
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
        "keypoint": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm")
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


def coco_evaluation(model, data_loader, iou_types=("bbox",), box_only=False, output_folder=None):
    """
    This function returns nothing on non-master processes (but still needs to be called).
    On master process, it returns the following:

    Returns:
       dict of dict: results[task][metric] is a float.
       list[dict]: one for each image, contains the outputs from the model.
       list[dict]: one for each instance, in the COCO evaluation format.
    """
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} images".format(len(dataset)))
    start_time = time.time()
    predictions = compute_on_dataset(model, data_loader)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    all_predictions = scatter_gather(predictions)
    if not is_main_process():
        return
    # concat the lists
    predictions = list(itertools.chain(*all_predictions))

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    # TODO(yuxin) test this codepath
    if box_only:
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(predictions, dataset, area=area, limit=limit)
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return

    logger.info("Preparing results for COCO format")
    coco_results = prepare_for_coco_evaluation(predictions)

    with tempfile.NamedTemporaryFile() as f:
        file_path = f.name
        if output_folder:
            file_path = os.path.join(output_folder, "coco_results.json")
        with open(file_path, "w") as f:
            json.dump(coco_results, f)

    results = COCOResults(*iou_types)
    if len(coco_results) == 0:
        return results.results, coco_results, predictions

    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        res = evaluate_predictions_on_coco(
            # TODO shouldn't use dataset.ds
            dataset.ds.coco,
            coco_results,
            file_path,
            iou_type,
        )
        results.update(res)
    results = results.results  # get the OrderedDict from COCOResults
    logger.info(results)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))

    return results, coco_results, predictions


def print_copypaste_format(results):
    """
    Print results in a format that's easy to copypaste to excel.

    Args:
        results: OrderedDict
    """
    assert isinstance(results, OrderedDict)  # unordered results cannot be properly printed
    logger = logging.getLogger(__name__)
    for task in ["bbox", "segm"]:
        if task not in results:
            continue
        res = results[task]
        logger.info("copypaste: Task: {}".format(task))
        logger.info("copypaste: " + ",".join([n for n in res.keys()]))
        logger.info("copypaste: " + ",".join(["{0:.4f}".format(v) for v in res.values()]))

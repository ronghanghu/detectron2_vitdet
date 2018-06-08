import torch

import torchvision
from torchvision.structures.bounding_box import BBox

import logging
import sys

from torch_detectron.helpers.config_utils import load_config

from tqdm import tqdm
def compute_on_dataset(model, data_loader, device):
    model.eval()
    results = []
    cpu_device = torch.device('cpu')
    for i, batch in tqdm(enumerate(data_loader)):
        images, targets = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        results.extend(output)
    return results


def prepare_for_coco_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction.bbox) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]['width']
        image_height = dataset.coco.imgs[original_id]['height']
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert('xywh')

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field('scores').tolist()
        labels = prediction.get_field('labels').tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend([{
            'image_id': original_id,
            'category_id': mapped_labels[k],
            'bbox': box,
            'score': scores[k]} for k, box in enumerate(boxes)])
    return coco_results


import time
def prepare_for_coco_segmentation(predictions, dataset):
    from .mask_rcnn import Masker
    import pycocotools.mask as mask_util
    import numpy as np
    masker = Masker(threshold=0.5, padding=1)
    assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction.bbox) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]['width']
        image_height = dataset.coco.imgs[original_id]['height']
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field('mask')
        # t = time.time()
        masks = masker(masks, prediction)
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field('scores').tolist()
        labels = prediction.get_field('labels').tolist()

        # rles = prediction.get_field('mask')

        rles = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], order='F'))[0] for mask in masks]
        for rle in rles:
            rle['counts'] = rle['counts'].decode('utf-8')

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend([{
            'image_id': original_id,
            'category_id': mapped_labels[k],
            'segmentation': rle,
            'score': scores[k]} for k, rle in enumerate(rles)])
    return coco_results


# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, area='all', limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        'all': 0,
        'small': 1,
        'medium': 2,
        'large': 3,
        '96-128': 4,
        '128-256': 5,
        '256-512': 6,
        '512-inf': 7}
    area_ranges = [
        [0**2, 1e5**2],    # all
        [0**2, 32**2],     # small
        [32**2, 96**2],    # medium
        [96**2, 1e5**2],   # large
        [96**2, 128**2],   # 96-128
        [128**2, 256**2],  # 128-256
        [256**2, 512**2],  # 256-512
        [512**2, 1e5**2]]  # 512-inf
    assert area in areas, 'Unknown area range: {}'.format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0
    from .box_ops import boxes_iou, boxes_area
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]['width']
        image_height = dataset.coco.imgs[original_id]['height']
        prediction = prediction.resize((image_width, image_height))

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj['bbox'] for obj in anno]
        gt_boxes = BBox(gt_boxes, (image_width, image_height), mode='xywh').convert('xyxy')

        gt_areas = boxes_area(gt_boxes.bbox)
        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += gt_boxes.bbox.shape[0]

        if gt_boxes.bbox.shape[0] == 0:
            continue

        if len(prediction.bbox) == 0:
            continue

        if limit is not None and prediction.bbox.shape[0] > limit:
            prediction = prediction[:limit]

        overlaps = boxes_iou(prediction.bbox, gt_boxes.bbox)

        _gt_overlaps = torch.zeros(gt_boxes.bbox.shape[0])
        for j in range(min(prediction.bbox.shape[0], gt_boxes.bbox.shape[0])):
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
    return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
            'gt_overlaps': gt_overlaps, 'num_pos': num_pos}


def evaluate_predictions_on_coco(coco_gt, coco_results,
        json_result_file, iou_type='bbox'):
    import json
    with open(json_result_file, 'w') as f:
        json.dump(coco_results, f)

    from pycocotools.cocoeval import COCOeval
    coco_dt = coco_gt.loadRes(str(json_result_file))
    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    # torch.save(coco_eval, join(args.save, 'computed_results.pth'))
    return coco_eval.summarize()


def inference(model,
        data_loader,
        iou_types=('bbox',),#'segm'),
        box_only=False,
        device=torch.device('cuda'),
        json_file=None):

    logger = logging.getLogger('__main__')
    dataset = data_loader.dataset
    predictions = compute_on_dataset(model, data_loader, device)
    if box_only:
        logger.info('Evaluating bbox proposals')
        results = evaluate_box_proposals(predictions, dataset, area='all', limit=1000)
        logger.info('AR@1000: {}'.format(results['ar'].item()))
        return
    logger.info('Preparing results for COCO format')
    coco_results = {}
    if 'bbox' in iou_types:
        logger.info('Preparing bbox results')
        coco_results['bbox'] = prepare_for_coco_detection(predictions, dataset)
    if 'segm' in iou_types:
        logger.info('Preparing segm results')
        coco_results['segm'] = prepare_for_coco_segmentation(predictions, dataset)

    logger.info('Evaluating predictions')
    for iou_type in iou_types:
        res = evaluate_predictions_on_coco(dataset.coco, coco_results[iou_type],
                'tmp_results/{}_{}.json'.format(json_file, iou_type), iou_type)
        logger.info(res)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Object Detection Inference')
    parser.add_argument('--config-file', default='/private/home/fmassa/github/detectron.pytorch/configs/faster_rcnn_r50_relaunch.py',
            metavar='FILE', help='path to config file')

    parser.add_argument('--checkpoint', default='/checkpoint02/fmassa/detectron_logs/faster_rcnn_R50_3970040/model_161249.pth', metavar='FILE', help='path to checkpoint file')

    args = parser.parse_args()
    config = load_config(args.config_file)
    config.distributed = False

    data_loader, data_loader_val = config.get_data_loader(config.distributed)
    model = config.get_model()

    checkpoint = torch.load(args.checkpoint)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    model = model.module

    inference(model, data_loader_val)

# -*- coding: utf-8 -*-
# File:

from detectron2.utils.env import setup_environment  # noqa F401 isort:skip
import numpy as np
from collections import defaultdict
import pycocotools.mask as mask_utils
import torch
import tqdm
from pycocotools.coco import COCO
from tabulate import tabulate
from torch.nn import functional as F

from detectron2.data import MetadataCatalog
from detectron2.modeling.roi_heads.paste_mask import (
    pad_masks,
    paste_mask_in_image,
    paste_masks_in_image_aligned,
    scale_boxes,
)
from detectron2.structures import Boxes, PolygonMasks
from detectron2.structures.masks import (
    batch_rasterize_full_image_polygons_within_box,
    rasterize_polygons_within_box,
)


"""
rasterize_polygons_within_box (used in training)
and
paste_masks_in_image (used in inference)
should be inverse operations to each other.

This script runs several implementation of the above two operations and prints
the reconstruction error.
"""


def polygons_to_full_image_bit_mask(polygons, img_h, img_w):
    rles = mask_utils.frPyObjects(polygons, img_h, img_w)
    rle = mask_utils.merge(rles)
    bit_mask = mask_utils.decode(rle)
    return bit_mask


def iou_between_full_image_bit_masks(a, b):
    def to_rle(mask):
        rle = mask_utils.encode(np.array(mask[:, :, None], order="F"))[0]
        return rle

    rle_a = to_rle(a)
    rle_b = to_rle(b)
    iscrowd = [int(False)]
    iou = mask_utils.iou([rle_a], [rle_b], iscrowd)
    return iou[0][0]


def rasterize_polygons_with_grid_sample(
    full_image_bit_mask, box, mask_size, half=0.5, threshold=0.5
):
    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]

    img_h, img_w = full_image_bit_mask.shape

    mask_y = np.arange(0.0, mask_size) + half  # mask y sample coords in [0.5, mask_size - 0.5]
    mask_x = np.arange(0.0, mask_size) + half  # mask x sample coords in [0.5, mask_size - 0.5]
    mask_y = (mask_y) / (mask_size) * (y1 - y0) + y0
    mask_x = (mask_x) / (mask_size) * (x1 - x0) + x0

    mask_x = (mask_x - 0.5) / (img_w - 1) * 2 + -1
    mask_y = (mask_y - 0.5) / (img_h - 1) * 2 + -1
    gy, gx = torch.meshgrid(torch.from_numpy(mask_y), torch.from_numpy(mask_x))
    ind = torch.stack([gx, gy], dim=-1).to(dtype=torch.float32)

    full_image_bit_mask = torch.from_numpy(full_image_bit_mask)
    mask = F.grid_sample(
        full_image_bit_mask[None, None, :, :].to(dtype=torch.float32), ind[None, :, :, :]
    )

    return mask[0, 0] >= threshold


def process_annotation(ann, mask_side_len=28):
    # Parse annotation data
    img_info = coco.loadImgs(ids=[ann["image_id"]])[0]
    height, width = img_info["height"], img_info["width"]
    gt_polygons = [np.array(p, dtype=np.float64) for p in ann["segmentation"]]
    gt_bbox = np.array(ann["bbox"])
    # Convert from XYWH to XYXY
    gt_bbox[2] += gt_bbox[0]
    gt_bbox[3] += gt_bbox[1]

    polygons_xmax = max(x.reshape(-1, 2)[:, 0].max() for x in gt_polygons)
    polygons_ymax = max(x.reshape(-1, 2)[:, 1].max() for x in gt_polygons)
    assert polygons_xmax <= gt_bbox[2] + 1e-8
    assert polygons_ymax <= gt_bbox[3] + 1e-8

    gt_full_image_bit_mask = polygons_to_full_image_bit_mask(gt_polygons, height, width)

    # Run rasterize ..
    gt_polygons_tensor = [torch.from_numpy(x) for x in gt_polygons]
    box_bitmasks = {
        "old": rasterize_polygons_within_box(gt_polygons_tensor, gt_bbox, mask_side_len),
        "gridsample": rasterize_polygons_with_grid_sample(
            gt_full_image_bit_mask, gt_bbox, mask_side_len
        ),
        "roialign": batch_rasterize_full_image_polygons_within_box(
            PolygonMasks([[x.tolist() for x in gt_polygons]]),
            torch.from_numpy(gt_bbox[None, :].astype("float32")),
            mask_side_len,
        )[0],
    }

    # Run paste ..
    results = defaultdict(dict)
    for k, box_bitmask in box_bitmasks.items():
        padded_bitmask, scale = pad_masks(box_bitmask[None, :, :], 1)
        scaled_boxes = scale_boxes(torch.from_numpy(gt_bbox[None, :]), scale)

        r = results[k]
        r["old"] = paste_mask_in_image(
            padded_bitmask[0], scaled_boxes[0], height, width, threshold=0.5
        )
        r["aligned"] = paste_masks_in_image_aligned(
            box_bitmask[None, None, :, :], Boxes(gt_bbox[None, :]), (height, width), threshold=0.5
        )[0, 0]

    table = []
    for rasterize_method, r in results.items():
        for paste_method, mask in r.items():
            iou = iou_between_full_image_bit_masks(gt_full_image_bit_mask, mask)
            table.append((rasterize_method, paste_method, iou))
    return table


if __name__ == "__main__":
    json_file = MetadataCatalog.get("coco_2017_val").json_file
    coco = COCO(json_file)
    anns = coco.loadAnns(coco.getAnnIds(iscrowd=False))  # avoid crowd annotations
    print("#anns", len(anns))

    selected_anns = anns[:1000]

    ious = []
    ious_s = []
    ious_m = []
    ious_l = []
    for ann in tqdm.tqdm(selected_anns):
        results = process_annotation(ann)
        data = [k[2] for k in results]
        ious.append(data)
        area = ann["area"]
        if area <= 32 ** 2:
            ious_s.append(data)
        elif area <= 96 ** 2:
            ious_m.append(data)
        else:
            ious_l.append(data)

    def summarize(ious):
        ious = np.array(ious)
        mean_ious = ious.mean(axis=0)

        table = []
        for res, iou in zip(results, mean_ious):
            table.append((res[0], res[1], iou))
        print(tabulate(table, headers=["rasterize", "paste", "iou"], tablefmt="simple"))

    print("All areas", len(ious))
    summarize(ious)
    print("-" * 80)
    print("Small", len(ious_s))
    summarize(ious_s)
    print("-" * 80)
    print("Medium", len(ious_m))
    summarize(ious_m)
    print("-" * 80)
    print("Large", len(ious_l))
    summarize(ious_l)

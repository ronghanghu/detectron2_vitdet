from datasets.pascal_voc import PascalVOC
from lib.collate_fn import Collator

from os.path import join

from torch.autograd import Variable
import torch.utils.data
from torchvision import transforms

from utils import cython_nms

import pickle

from tqdm import tqdm

import numpy as np
import json
from pycocotools.cocoeval import COCOeval

import pycocotools.mask as mask_util

from torch import nn
from lib.utils.box import clip_boxes_to_image
from lib.utils.box_coder import BoxCoder


from PIL import Image


# TODO maybe remove im_shape if we use normalized boxes
def apply_box_deltas(boxes, box_deltas, im_shape):
    #pred_boxes = bbox_transform(
    #    boxes, box_deltas, (10., 10., 5., 5.)  # cfg.MODEL.BBOX_REG_WEIGHTS
    #)
    pred_boxes = BoxCoder((10., 10., 5., 5.)).decode(box_deltas, boxes)
    height, width = im_shape
    pred_boxes = clip_boxes_to_image(pred_boxes, height, width)
    
    return pred_boxes

def box_results_with_nms_and_limit(scores, boxes, score_thresh=0.05, nms=0.5, detections_per_img=100):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).
    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.
    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = scores.shape[1]
    cls_boxes = [[] for _ in range(num_classes)]
    perm_idx = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > score_thresh)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        """
        if cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:"""
        keep = cython_nms.nms(dets_j, nms)
        nms_dets = dets_j[keep, :]
        """
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        """
        cls_boxes[j] = nms_dets
        perm_idx[j] = inds[keep]

    # Limit to max_per_image detections **over all classes**
    if detections_per_img > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > detections_per_img:
            # TODO can optimize using torch kthvalue
            image_thresh = np.sort(image_scores)[-detections_per_img]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]
                perm_idx[j] = perm_idx[j][keep]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes, perm_idx

def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')

def expand_boxes(boxes, scale):
    """Expand an array of boxes by a given scale."""
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp

def paste_mask_in_image(mask, box, im_h, im_w):

    if True:
        M = 28
        scale = (M + 2.0) / M
        padded_mask = torch.zeros((M + 2, M + 2), dtype=torch.float32)
        padded_mask[1:-1, 1:-1] = mask
        mask = padded_mask
        box = expand_boxes(box[None], scale)[0]
    box = box.astype(np.int32)
        

    TO_REMOVE = 1
    w = box[2] - box[0] + TO_REMOVE
    h = box[3] - box[1] + TO_REMOVE
    w = max(w, 1)
    h = max(h, 1)

    mask = Image.fromarray(mask.numpy())
    mask = mask.resize((w, h), resample=Image.BILINEAR)
    mask = np.array(mask, copy=False)

    THRESH_BINARIZE = 0.5
    mask = np.array(mask > THRESH_BINARIZE, dtype=np.uint8)
    mask = torch.from_numpy(mask)

    im_mask = torch.zeros((im_h, im_w), dtype=torch.uint8)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]):(y_1 - box[1]),
        (x_0 - box[0]):(x_1 - box[0])
    ]
    return im_mask


def evaluate(model, data_loader, args):
    results = []
    mask_results = []
    orig_box = []
    if torch.cuda.is_available():
        model.cuda()
    """
    for i, batch in tqdm(enumerate(data_loader)):
        imgs, boxes, labels, im_shape, img_idx = batch

        if False:
            # TODO attention because this is on a batch of images!!!
            DEDUP_BOXES = 1 / 16.
            # v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            v = np.array([1, 1e3, 1e6, 1e9])
            # hashes = np.round(inputs['rois'] * DEDUP_BOXES).dot(v)
            hashes = np.round(boxes.numpy() * DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(
                hashes, return_index=True, return_inverse=True
            )
            # inputs['rois'] = inputs['rois'][index, :]
            boxes = boxes.numpy()
            boxes = boxes[index, :]
            boxes = torch.from_numpy(boxes).pin_memory()

        if torch.cuda.is_available():
            imgs = imgs.cuda(non_blocking=True)
            boxes = boxes.cuda(non_blocking=True)
            im_shape = im_shape.cuda(non_blocking=True)


        with torch.no_grad():
            # scores, boxes, box_deltas = model(imgs, im_shape)
            scores, box_deltas, rpn_scores, rpn_box_deltas, boxes, all_anchors, img_idx_proposal, inds_inside, idxs, masks = model(imgs, im_shape)
            #scores, box_deltas, rpn_scores, rpn_box_deltas, boxes, all_anchors, img_idx_proposal, inds_inside, idxs, features = model(imgs, im_shape)
            scores = torch.nn.functional.softmax(scores, -1)
            box_deltas = box_deltas
            masks = masks.sigmoid()

        #scores = scores.cpu().numpy()
        #box_deltas = box_deltas.cpu().numpy()
        #boxes = boxes.cpu().numpy()
        # TODO this should be on a per-image basis, not on the batched version...
        for im in range(imgs.shape[0]):
            # pred_boxes = apply_box_deltas(boxes[img_idx[im]], box_deltas[img_idx[im]], im_shape[im])
            pred_boxes = apply_box_deltas(boxes, box_deltas, im_shape[im])
        
            if False:
                scores = scores[inv_index, :]
                pred_boxes = pred_boxes[inv_index, :]

            scores = scores.cpu().numpy()
            pred_boxes = pred_boxes.cpu().numpy()
            # TODO unfuse scores and pred_boxes?
            # TODO make sure that we always have num_classes lists
            # _, _, im_results = box_results_with_nms_and_limit(scores[img_idx], pred_boxes)
            _, final_boxes, im_results, perm_idx = box_results_with_nms_and_limit(scores, pred_boxes)
            height, width = im_shape[im].tolist()
            # normalize boxes
            for l in im_results:
                if len(l) == 0:
                    continue
                l[:, 0] /= width
                l[:, 1] /= height
                l[:, 2] /= width
                l[:, 3] /= height

            results.extend([im_results])

            selected_masks = [[] for _ in range(len(im_results))]
            selected_orig_box = [[] for _ in range(len(im_results))]
            masks = masks.cpu()
            counter = 0

            boxes[:, 0] /= width
            boxes[:, 1] /= height
            boxes[:, 2] /= width
            boxes[:, 3] /= height

            for i in range(len(im_results)):
                if len(im_results[i]) == 0:
                    continue
                selected_masks[i] = masks[perm_idx[i], i]
                selected_orig_box[i] = boxes[perm_idx[i]]
            mask_results.extend([selected_masks])
            orig_box.extend([selected_orig_box])

    # transpose results, could also just be list(zip(*results))
    results = list(map(list, zip(*results)))
    mask_results = list(map(list, zip(*mask_results)))
    orig_box = list(map(list, zip(*orig_box)))
    # TODO save the list
    torch.save(results, join(args.save, 'results.pth'))
    torch.save(mask_results, join(args.save, 'mask_results_v3.pth'))
    torch.save(orig_box, join(args.save, 'orig_box_results_v3.pth'))
    """

    results = torch.load(join(args.save, 'results.pth'))
    mask_results = torch.load(join(args.save, 'mask_results_v3.pth'))
    orig_box = torch.load(join(args.save, 'orig_box_results_v3.pth'))
    # results is a [num_classes][num_images][array(dets, 5)] list
    to_json_results = []
    to_json_results_mask = []
    dataset = data_loader.dataset
    for cl, per_cat_results in enumerate(results):
        # background
        if cl == 0:
            continue
        cat_id = dataset.contiguous_category_id_to_json_id[cl]
        for im, dets in enumerate(per_cat_results):
            image_id = dataset.id_to_img_map[im]
            if isinstance(dets, list) and len(dets) == 0:
                continue
            dets = dets.astype(np.float)
            scores = dets[:, -1]
            height = dataset.coco.imgs[image_id]['height']
            width = dataset.coco.imgs[image_id]['width']
            boxes = dets[:, :4]
            boxes[:, 0] *= width
            boxes[:, 1] *= height
            boxes[:, 2] *= width
            boxes[:, 3] *= height
            xywh_dets = xyxy_to_xywh(boxes)
            xs, ys, ws, hs = xywh_dets.T
            to_json_results.extend(
                        [{'image_id': image_id,
                          'category_id': cat_id,
                          'bbox': [xs[k], ys[k], ws[k], hs[k]],
                          'score': scores[k]} for k in range(dets.shape[0])])
            
            masks = mask_results[cl][im]
            # boxes = boxes.astype(np.int32)
            orig_b = orig_box[cl][im]
            #print('orig_b', orig_b)
            #print('masks', masks)
            if len(orig_b) > 0:
                orig_b[:, 0] *= width
                orig_b[:, 1] *= height
                orig_b[:, 2] *= width
                orig_b[:, 3] *= height
            img_masks = [paste_mask_in_image(m, orig_b[iidx], height, width).numpy() for iidx, m in enumerate(masks)]
            rles = [mask_util.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0] for im_mask in img_masks]
            for rle in rles:
                rle['counts'] = rle['counts'].decode('utf-8')
            to_json_results_mask.extend(
                        [{'image_id': image_id,
                          'category_id': cat_id,
                          'segmentation': rles[k],
                          'score': scores[k]} for k in range(dets.shape[0])])

    """
    res_file = join(args.save, 'json_results.json')
    with open(res_file, 'w') as f:
        json.dump(to_json_results, f)
    """

    res_file_mask = join(args.save, 'json_results_mask_v3.json')
    with open(res_file_mask, 'w') as f:
        json.dump(to_json_results_mask, f)

    """
    # res_file = '/checkpoint02/fmassa/bottleneck_debug/json_results.json'
    coco_dt = dataset.coco.loadRes(str(res_file))
    coco_eval = COCOeval(dataset.coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    torch.save(coco_eval, join(args.save, 'computed_results.pth'))
    coco_eval.summarize()
    """

    coco_dt = dataset.coco.loadRes(str(res_file_mask))
    coco_eval_m = COCOeval(dataset.coco, coco_dt, 'segm')
    coco_eval_m.evaluate()
    coco_eval_m.accumulate()
    torch.save(coco_eval_m, join(args.save, 'computed_results_mask_v3.pth'))
    coco_eval_m.summarize()


    return coco_eval.stats, coco_eval_m.stats

if __name__ == '__main__':

    import torch.utils.data
    import torchvision.transforms as T

    from datasets.coco import COCODataset
    from lib.collate_fn import Collator
    from lib.models_caffe2 import load_resnet50_model
    from lib.mrcnn import C2MRCNNResNetFPN
    from lib.data_transforms.transforms import Resize
    from lib.data_transforms.transform_dataset import TransformDataset

    print('Start evaluation script')
    ann_file = '/private/home/fmassa/coco_trainval2017/annotations/instances_val2017_mod.json'
    data_dir = '/datasets01/COCO/060817/val2014/'

    # transforms = T.Compose([T.Resize(800), T.ToTensor(), T.Lambda(lambda x: (x[[2,1,0]] * 255) - torch.Tensor([102.9801, 115.9465, 122.7717]).view(3,1,1))])
    # transforms = T.Compose([T.ToTensor(), 
    #     T.Lambda(lambda x: (x[[2,1,0]] * 255) - torch.Tensor([102.9801, 115.9465, 122.7717]).view(3,1,1))])
    normalize = T.Lambda(lambda x: (x[[2,1,0]] * 255) - torch.Tensor([102.9801, 115.9465, 122.7717]).view(3,1,1))

    #normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
    #                        std=[0.229, 0.224, 0.225])

    transforms = T.Compose([T.ToTensor(), normalize])
    def tt(args):
        image = args[0]
        image = transforms(image)
        return (image,) + args[1:]
    # dataset = COCODataset(ann_file, data_dir, transform=transforms)
    dataset = COCODataset(ann_file, data_dir)

    ttt = T.Compose([Resize(800, 1333), tt])

    dataset = TransformDataset(dataset, ttt)

    collate_fn = Collator()

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = C2MRCNNResNetFPN('/private/home/fmassa/github/detectron.pytorch/torch_detectron/mrcnn-fpn50.pkl')
    # model.load_state_dict(torch.load('/checkpoint02/fmassa/detectron_v2_faster_rcnn_R50.pth')['model'])

    # model = fastrcnn_resnet18(num_classes=81)
    # model.load_state_dict(torch.load('/checkpoint02/fmassa/detectron_v0.pth')['model'])
    model.eval()

    class Args(object):
        def __init__(self):
            # self.save = '/checkpoint02/fmassa/bottleneck_debug'
            self.save = ''

    evaluate(model, data_loader, Args())


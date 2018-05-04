from datasets.pascal_voc import PascalVOC
from lib.collate_fn import Collator
from lib.transforms import VOCAnnotations

from torch.autograd import Variable
import torch.utils.data
from torchvision import transforms

from utils import cython_nms

import pickle

from tqdm import tqdm

import numpy as np
import json
from pycocotools.cocoeval import COCOeval

from torch import nn

# TODO change the width and remove the +1
def bbox_transform(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
    """Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas. See bbox_transform_inv for a
    description of the weights argument.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    BBOX_XFORM_CLIP = np.log(1000. / 16.)
    dw = np.minimum(dw, BBOX_XFORM_CLIP)
    dh = np.minimum(dh, BBOX_XFORM_CLIP)

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


def clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes
    has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 4 == 0, \
        'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(
        boxes.shape[1]
    )
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

# TODO maybe remove im_shape if we use normalized boxes
def apply_box_deltas(boxes, box_deltas, im_shape):
    pred_boxes = bbox_transform(
        boxes, box_deltas, (10., 10., 5., 5.)  # cfg.MODEL.BBOX_REG_WEIGHTS
    )
    pred_boxes = clip_tiled_boxes(pred_boxes, im_shape)
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

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes

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


def evaluate(model, data_loader):
    results = []
    if torch.cuda.is_available():
        model.cuda()
    
    for i, batch in tqdm(enumerate(data_loader)):
        imgs, boxes, labels, im_shape, img_idx = batch

        if True:
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

        imgs = torch.autograd.Variable(imgs)
        boxes = torch.autograd.Variable(boxes)

        """
        img_idx_v2 = torch.LongTensor(img_idx[-1].stop)
        for i, s in enumerate(img_idx):
            img_idx_v2[s].fill_(i)
        """
        with torch.no_grad():
            # scores, box_deltas = model(imgs, boxes, img_idx_v2)
            # scores, box_deltas = model(imgs, boxes, boxes[:, 0].data.float() * 0)
            scores, box_deltas = model(imgs, boxes, boxes[:, 0].float() * 0)

        boxes = boxes.data
        scores = scores.cpu().numpy()
        box_deltas = box_deltas.cpu().numpy()
        boxes = boxes.cpu().numpy()
        # TODO this should be on a per-image basis, not on the batched version...
        for im in range(imgs.shape[0]):
            # pred_boxes = apply_box_deltas(boxes[img_idx[im]], box_deltas[img_idx[im]], im_shape[im])
            pred_boxes = apply_box_deltas(boxes, box_deltas, im_shape[im])
        
            if True:
                scores = scores[inv_index, :]
                pred_boxes = pred_boxes[inv_index, :]

            # TODO unfuse scores and pred_boxes?
            # TODO make sure that we always have num_classes lists
            # _, _, im_results = box_results_with_nms_and_limit(scores[img_idx], pred_boxes)
            _, _, im_results = box_results_with_nms_and_limit(scores, pred_boxes)
            results.extend([im_results])
    # transpose results, could also just be list(zip(*results))
    results = list(map(list, zip(*results)))
    # TODO save the list
    torch.save(results, 'results.pth')
    
    # results = torch.load('results.pth')
    # results is a [num_classes][num_images][array(dets, 5)] list
    to_json_results = []
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
            xywh_dets = xyxy_to_xywh(dets[:, 0:4])
            xs, ys, ws, hs = xywh_dets.T
            to_json_results.extend(
                        [{'image_id': image_id,
                          'category_id': cat_id,
                          'bbox': [xs[k], ys[k], ws[k], hs[k]],
                          'score': scores[k]} for k in range(dets.shape[0])])
    
    res_file = 'json_results.json'                
    with open(res_file, 'w') as f:
        json.dump(to_json_results, f)
    
    # res_file = '/private/home/fmassa/bbox_coco_2014_minival_results.json'
    coco_dt = dataset.coco.loadRes(str(res_file))
    coco_eval = COCOeval(dataset.coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    torch.save(coco_eval, 'computed_results.pth')
    coco_eval.summarize()


class ProposalDataset(object):
    def __init__(self, dataset, proposal_file):
        self.dataset = dataset
        with open(proposal_file, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')

        proposals_map_to_id = {v: k for k, v in enumerate(self.data['ids'])}
        self.img_id_map_to_proposal_id = {k: proposals_map_to_id[v]
                for k, v in enumerate(dataset.img_ids)}

    def __getitem__(self, idx):
        item = self.dataset[idx]
        proposal_id = self.img_id_map_to_proposal_id[idx]
        boxes = torch.from_numpy(self.data['boxes'][proposal_id])
        
        # removing the gt boxes for now, might want to keep it and perform bbox assignment and extend labels
        new_item = (item[0], boxes) + item[2:]

        return new_item

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name in self.dataset.__dict__:
            return getattr(self.dataset, name)
        return object.__getattr__(self, name)

if __name__ == '__main__':

    import torch.utils.data
    import torchvision.transforms as T

    from datasets.coco import COCODataset
    from lib.collate_fn import Collator
    from lib.models_caffe2 import load_resnet50_model
    from lib.models.fastrcnn import fastrcnn_resnet18, fastrcnn_resnet50

    ann_file = '/private/home/fmassa/coco_trainval2017/annotations/instances_val2017.json'
    data_dir = '/datasets01/COCO/060817/val2014/'


    # transforms = T.Compose([T.Resize(800), T.ToTensor(), T.Lambda(lambda x: (x[[2,1,0]] * 255) - torch.Tensor([102.9801, 115.9465, 122.7717]).view(3,1,1))])
    transforms = T.Compose([T.ToTensor(), 
        T.Lambda(lambda x: (x[[2,1,0]] * 255) - torch.Tensor([102.9801, 115.9465, 122.7717]).view(3,1,1))])

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    transforms = T.Compose([T.ToTensor(), normalize])

    dataset = COCODataset(ann_file, data_dir, transform=transforms)

    collate_fn = Collator()

    dataset = ProposalDataset(dataset, '/private/home/fmassa/rpn_proposals.pkl')

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # model = load_resnet50_model(file_path='/private/home/fmassa/model_final.pkl')

    model = fastrcnn_resnet18(num_classes=81)
    model.load_state_dict(torch.load('/checkpoint02/fmassa/detectron_v0.pth')['model'])

    evaluate(model, data_loader)


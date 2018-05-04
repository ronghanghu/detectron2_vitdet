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

from torch import nn
from lib.utils.box import clip_boxes_to_image
from lib.utils.box_coder import BoxCoder


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


def evaluate(model, data_loader, args):
    results = []
    if torch.cuda.is_available():
        model.cuda()
    
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
            rpn_scores, rpn_box_deltas, all_anchors, img_idx_proposal, inds_inside = model(imgs, im_shape)
            boxes, selected_scores, idxs = model.selector.select_top(all_anchors, torch.nn.functional.sigmoid(rpn_scores), rpn_box_deltas, im_shape)
            

        #scores = scores.cpu().numpy()
        #box_deltas = box_deltas.cpu().numpy()
        #boxes = boxes.cpu().numpy()
        # TODO this should be on a per-image basis, not on the batched version...
        for im in range(imgs.shape[0]):
            selected_scores = torch.cat(selected_scores, dim=0)
            selected_scores = selected_scores.cpu().numpy()
            boxes = boxes.cpu().numpy()
            height, width = im_shape[im].tolist()
            # normalize boxes
            boxes[:, 0] /= width
            boxes[:, 1] /= height
            boxes[:, 2] /= width
            boxes[:, 3] /= height

            results.append((selected_scores, boxes))
    # transpose results, could also just be list(zip(*results))
    # results = list(map(list, zip(*results)))
    # TODO save the list
    torch.save(results, join(args.save, 'results.pth'))
    
    # results = torch.load('results.pth')
    # results is a [num_classes][num_images][array(dets, 5)] list
    to_json_results = []
    dataset = data_loader.dataset
    cat_id = 1
    for im, dets in enumerate(results):
        image_id = dataset.id_to_img_map[im]
        scores, boxes = dets
        height = dataset.coco.imgs[image_id]['height']
        width = dataset.coco.imgs[image_id]['width']
        boxes[:, 0] *= width
        boxes[:, 1] *= height
        boxes[:, 2] *= width
        boxes[:, 3] *= height
        xywh_dets = xyxy_to_xywh(boxes)
        xs, ys, ws, hs = xywh_dets.T
        xs = xs.tolist()
        ys = ys.tolist()
        ws = ws.tolist()
        hs = hs.tolist()
        scores = scores.tolist()
        to_json_results.extend(
                    [{'image_id': image_id,
                      'category_id': cat_id,
                      'bbox': [xs[k], ys[k], ws[k], hs[k]],
                      'score': scores[k]} for k in range(boxes.shape[0])])

    """
    to_json_results = []
    with open('/private/home/fmassa/rpn_proposals.pkl', 'rb') as f:
         data = pickle.load(f, encoding='latin1')
    for i, image_id in enumerate(data['ids']):
        scores = data['scores'][i]
        boxes = data['boxes'][i]

        height = dataset.coco.imgs[image_id]['height']
        width = dataset.coco.imgs[image_id]['width']
        xywh_dets = xyxy_to_xywh(boxes)
        xs, ys, ws, hs = xywh_dets.T
        xs = xs.tolist()
        ys = ys.tolist()
        ws = ws.tolist()
        hs = hs.tolist()
        scores = scores.tolist()
        to_json_results.extend(
                    [{'image_id': image_id,
                      'category_id': cat_id,
                      'bbox': [xs[k], ys[k], ws[k], hs[k]],
                      'score': scores[k]} for k in range(boxes.shape[0])])
    """


    res_file = join(args.save, 'json_results.json')
    print(type(to_json_results[0]['score']))
    with open(res_file, 'w') as f:
        json.dump(to_json_results, f)
    

    # res_file = '/private/home/fmassa/bbox_coco_2014_minival_results.json'
    coco_dt = dataset.coco.loadRes(str(res_file))
    coco_eval = COCOeval(dataset.coco, coco_dt, 'bbox')
    coco_eval.params.maxDets = [1000]
    coco_eval.params.useCats = 0
    coco_eval.evaluate()
    coco_eval.accumulate()
    torch.save(coco_eval, join(args.save, 'computed_results.pth'))
    if True:
        areaRng = 'all'; maxDets = 1000
        p = coco_eval.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        s = coco_eval.eval['recall']
        ss = s[:,:,aind,mind]
        print(np.mean(ss[ss>-1]))
        return np.mean(ss[ss>-1])
    else:
        coco_eval.summarize()

if __name__ == '__main__':

    import torch.utils.data
    import torchvision.transforms as T

    from datasets.coco import COCODataset
    from lib.collate_fn import Collator
    from lib.models_caffe2 import load_resnet50_model
    from lib.faster_rcnn import fasterrcnn_resnet18, C2ResNetFasterRCNN
    from lib.data_transforms.transforms import Resize
    from lib.data_transforms.transform_dataset import TransformDataset

    ann_file = '/private/home/fmassa/coco_trainval2017/annotations/instances_val2017.json'
    data_dir = '/datasets01/COCO/060817/val2014/'

    # transforms = T.Compose([T.Resize(800), T.ToTensor(), T.Lambda(lambda x: (x[[2,1,0]] * 255) - torch.Tensor([102.9801, 115.9465, 122.7717]).view(3,1,1))])
    transforms = T.Compose([T.ToTensor(), 
        T.Lambda(lambda x: (x[[2,1,0]] * 255) - torch.Tensor([102.9801, 115.9465, 122.7717]).view(3,1,1))])

    # normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225])

    # transforms = T.Compose([T.ToTensor(), normalize])
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

    # model = load_resnet50_model(file_path='/private/home/fmassa/model_final.pkl')
    # model = fasterrcnn_resnet18(num_classes=81).cuda()
    model = C2ResNetFasterRCNN('/private/home/fmassa/model_final_faster_rcnn.pkl').cuda()
    model.load_state_dict(torch.load('/checkpoint02/fmassa/detectron_v0_rpn_with_resize.pth')['model'])

    # model = fastrcnn_resnet18(num_classes=81)
    # model.load_state_dict(torch.load('/checkpoint02/fmassa/detectron_v0.pth')['model'])

    evaluate(model, data_loader)


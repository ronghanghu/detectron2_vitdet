import math

import torch
from torch import nn

from nms import nms as box_nms
from box_coder import BoxCoder
import box as box_utils

# TODO maybe remove im_shape if we use normalized boxes
def apply_box_deltas(boxes, box_deltas, im_shape):
    pred_boxes = BoxCoder((10., 10., 5., 5.)).decode(box_deltas, boxes)
    height, width = im_shape
    pred_boxes = box_utils.clip_boxes_to_image(pred_boxes, height, width)
    
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
    cls_scores = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = (scores[:, j] > score_thresh).nonzero()
        inds = inds.squeeze(1) if inds.dim() == 2 else inds
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        keep = box_nms(boxes_j, scores_j, nms)
        cls_boxes[j] = boxes_j[keep]
        cls_scores[j] = scores_j[keep]

    # Limit to max_per_image detections **over all classes**
    if detections_per_img > 0:
        image_scores = torch.cat(
            [cls_scores[j] for j in range(1, num_classes)], dim=0
        )
        if len(image_scores) > detections_per_img:
            # TODO can optimize using torch kthvalue
            # image_thresh = np.sort(image_scores)[-detections_per_img]
            image_thresh, _ = torch.kthvalue(image_scores,
                    image_scores.shape[0] - detections_per_img)
            for j in range(1, num_classes):
                keep = (cls_scores[j] >= image_thresh).nonzero()
                keep = keep.squeeze(1) if keep.dim() == 2 else keep
                cls_boxes[j] = cls_boxes[j][keep]
                cls_scores[j] = cls_scores[j][keep]


    return cls_scores, cls_boxes


class DetectionModel(nn.Module):
    def __init__(self, min_size=800, max_size=1300, class_names=None, **kwargs):
        super(DetectionModel, self).__init__(**kwargs)
        self.min_size = float(min_size)
        self.max_size = float(max_size)
        self.class_names = class_names

    def _scale_image(self, image):
        image = image[None] if image.dim() == 3 else image
        h, w = image.shape[2:]
        min_orig_size = min(h, w)
        max_orig_size = max(h, w)
        ratio = self.min_size / min_orig_size
        if max_orig_size * ratio > self.max_size:
            ratio = self.max_size / max_orig_size
        sizes = [int(math.ceil(h * ratio)), int(math.ceil(w * ratio))]
        with torch.no_grad():
            image = nn.functional.upsample(image, size=sizes,
                    mode='bilinear', align_corners=False)
        return image[0]


    def _prepare_images(self, images):
        images = [self._scale_image(image) for image in images]

        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))

        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).zero_()
        for img, pad_img in zip(images, batched_imgs):
            img = self._preprocess_image(img)
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        img_sizes = torch.tensor([img.shape[1:] for img in images], dtype=batched_imgs.dtype)
        return batched_imgs, img_sizes


    def predict(self, images):
        """
        images: list of 3xHxW image tensors, where each tensor can
        have different sizes
        The tensors should be in range 0-1, in RGB format
        """
        if not isinstance(images, (list, tuple)):
            images = [images]

        original_sizes = torch.tensor([image.shape[1:] for image in images], dtype=images[0].dtype)

        imgs, img_sizes = self._prepare_images(images)
        with torch.no_grad():
            scores, box_deltas, _, _, boxes, _, _, _, img_idxs_mask = self(imgs, img_sizes)
            scores = torch.nn.functional.softmax(scores, -1)

        final_boxes = []
        final_scores = []
        final_names = []
        for img_idx, (img_size, orig_size) in enumerate(zip(img_sizes, original_sizes)):
            idx = img_idxs_mask == img_idx
            boxes_per_img = boxes[idx]
            box_deltas_per_img = box_deltas[idx]
            scores_per_img = scores[idx]

            pred_boxes = apply_box_deltas(boxes_per_img, box_deltas_per_img, img_size)
            pred_boxes = pred_boxes.cpu()
            scores_per_img = scores_per_img.cpu()

            selected_scores, selected_boxes = box_results_with_nms_and_limit(scores_per_img, pred_boxes)

            height, width = img_size
            orig_height, orig_width = orig_size
            ratio = float(orig_width) / width
            selected_boxes = [box * ratio if len(box) else box for box in selected_boxes]

            class_idx = [i for i, cls in enumerate(selected_scores) for _ in range(len(cls))]
            class_names = [self.class_names[i] for i in class_idx]

            selected_boxes = torch.cat([bbox for bbox in selected_boxes if len(bbox)])
            selected_scores = torch.cat([s for s in selected_scores if len(s)])

            final_boxes.append(selected_boxes)
            final_scores.append(selected_scores)
            final_names.append(class_names)

        return final_scores, final_boxes, final_names


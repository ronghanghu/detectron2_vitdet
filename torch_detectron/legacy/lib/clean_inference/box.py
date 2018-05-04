import torch
from torch.jit import compile

import numpy as np


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py

# for a 80000x4 and 30x4 inputs, here are the numbers
# cython implementation: 7.05 ms ± 62 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# this implementation cuda: 1.2 ms ± 997 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# this with compilation in cuda: 383 µs ± 483 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# and gives same results

# WARN different than Ross' implementation. No +1 in area
@compile(nderivs=0)
def box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt + 1).clamp(min=0)      # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)  # [N,]
    area2 = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)  # [M,]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

@compile(nderivs=0)
def boxes_area(box):
    TO_REMOVE = 1
    area = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
    return area


def clip_boxes_to_image(boxes, height, width):
    # boxes = boxes.clamp(min=0)
    # boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(max=width)
    # boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(max=height)

    fact = 1  # TODO REMOVE
    num_boxes = boxes.shape[0]
    b1 = boxes[:, 0::2].clamp(min=0, max=width-fact)
    b2 = boxes[:, 1::2].clamp(min=0, max=height-fact)
    boxes = torch.stack((b1, b2), 2).view(num_boxes, -1)

    # The below operations modify ops that for backward break things
    # a solution would be to use torch.no_grad, but let's leave it like that
    """
    boxes[:, 0::4] = boxes[:, 0::4].clamp(max=width-fact)
    boxes[:, 1::4] = boxes[:, 1::4].clamp(max=height-fact)
    boxes[:, 2::4] = boxes[:, 2::4].clamp(max=width-fact)
    boxes[:, 3::4] = boxes[:, 3::4].clamp(max=height-fact)
    """
    return boxes

def filter_small_boxes(boxes, min_size):
    """Keep boxes with width and height both greater than min_size."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = torch.nonzero((w > min_size) & (h > min_size)).squeeze(1)
    return keep

def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    boxes = boxes.numpy()
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return torch.from_numpy(np.sort(index))


# TODO this is wrong, only return a subset of the nms boxes
def nms(boxes, scores, nms_thresh):
    _, order = scores.sort(dim=0, descending=True)
    boxes_ = boxes[order]
    iou = box_iou(boxes_, boxes_)
    iou = torch.triu(iou >= nms_thresh, diagonal=1).int()
    # taken = (iou.sum(0) == 0).int()

    # blacklisted = ((iou * taken[:, None]).sum(0) > 0).int()

    # now need to handle the case where there are higher score boxes that got dropped
    # iou *= blacklisted[:, None]
    iou = iou.sum(0) == 0

    inverse_order = order.new(order.shape)
    inverse_order[order] = torch.arange(0, order.shape[0], dtype=order.dtype)

    keep = iou[inverse_order].nonzero().squeeze(1)
    return keep


if __name__ == '__main__':
    from utils.cython_nms import nms as cython_nms
    a = torch.rand(100, 4)
    b = torch.rand(100)
    aa = torch.cat((a, b[:, None]), 1).numpy()
    keep1 = nms(a, b, 0.3).long()
    keep2 = torch.from_numpy(cython_nms(aa, 0.3)).long()
    print(keep1.equal(keep2))
    print(keep1)
    print(keep2)
    from IPython import embed; embed()

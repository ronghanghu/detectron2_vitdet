import torch


def boxes_area(box):
    """
    Arguments:
        box: tensor
    """
    TO_REMOVE = 1
    area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
    return area


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxes_iou(box1, box2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    assert box1.size == box2.size
    box1, box2 = box1.bbox, box2.bbox
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0] + TO_REMOVE) * (
        box1[:, 3] - box1[:, 1] + TO_REMOVE
    )  # [N,]
    area2 = (box2[:, 2] - box2[:, 0] + TO_REMOVE) * (
        box2[:, 3] - box2[:, 1] + TO_REMOVE
    )  # [M,]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

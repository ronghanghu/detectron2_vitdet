import torch
from torch.jit import compile


def boxes_area(box):
    TO_REMOVE = 1
    area = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
    return area


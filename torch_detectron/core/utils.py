"""
Miscellaneous utility functions
"""

import torch

from ..structures.bounding_box import BBox


def nonzero(tensor):
    """
    Equivalent to numpy.nonzero(). The difference torch.nonzero()
    is that it returns a tuple of 1d tensors, and not a 2d tensor.
    This is more convenient in a few cases.
    This should maybe be sent to core pytorch
    """
    result = tensor.nonzero()
    if result.numel() > 0:
        return torch.unbind(result, 1)
    return (result,) * tensor.dim()


# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1./9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_bbox(bboxes):
    """
    Concatenates a list of BBox (having the same image size) into a
    single BBox

    Arguments:
        bboxes (list of BBox)
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BBox) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BBox(cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes

def split_bbox(bbox, split_size_or_sections):
    assert isinstance(bbox, BBox)

    boxes = bbox.bbox.split(split_size_or_sections, dim=0)
    size = bbox.size
    mode = bbox.mode
    fields = bbox.fields()
    fields_data = {field:bbox.get_field(field).split(split_size_or_sections)
            for field in fields}
    bboxes = [BBox(box, size, mode) for box in boxes]
    for i, box in enumerate(bboxes):
        [box.add_field(field, field_data[i])
                for field, field_data in fields_data.items()]

    return bboxes

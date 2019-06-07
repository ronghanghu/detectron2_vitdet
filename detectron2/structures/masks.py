import copy
import numpy as np
import pycocotools.mask as mask_utils
import torch

from detectron2.layers.roi_align import ROIAlign


def rasterize_polygons_within_box(polygons, box, mask_size):
    """
    Rasterize the polygons into a mask image and
    crop the mask content in the given box.
    The cropped mask is resized to (mask_size, mask_size).

    This function is used when generating training targets for mask head in Mask R-CNN.
    Given original ground-truth masks for an image, new ground-truth mask
    training targets in the size of `mask_size x mask_size`
    must be provided for each predicted box. This function will be called to
    produce such targets.

    Args:
        polygons (list[Tensor[float]]): a list of polygons, which represents an instance.
        box: 4 elements
        mask_size (int):

    Returns:
        Tensor: ByteTensor of shape (mask_size, mask_size)
    """
    # 1. Shift the polygons w.r.t the boxes
    w, h = box[2] - box[0], box[3] - box[1]

    polygons = copy.deepcopy(polygons)
    for p in polygons:
        p[0::2] = p[0::2] - box[0]
        p[1::2] = p[1::2] - box[1]

    # 2. Rescale the polygons to the new box size
    ratio_h = mask_size / max(h, 0.1)
    ratio_w = mask_size / max(w, 0.1)

    if ratio_h == ratio_w:
        for p in polygons:
            p *= ratio_h
    else:
        for p in polygons:
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h

    # 3. Rasterize the polygons with coco api
    rles = mask_utils.frPyObjects(polygons, mask_size, mask_size)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    mask = torch.from_numpy(mask)
    return mask


def batch_rasterize_polygons_within_box(masks, boxes, mask_side_len):
    """
    Batched version of :func:`rasterize_polygons_within_box`.

    Args:
        masks (PolygonMasks): store N masks for an image in polygon format.
        boxes (Tensor): store N boxes corresponding to the masks.
        mask_size (int): the size of the rasterized mask.

    Returns:
        Tensor: A byte tensor of shape (N, mask_size, mask_size), where
            N is the number of predicted boxes for this image.
    """
    device = boxes.device
    # Put boxes on the CPU, as the representation for masks is not efficient
    # GPU-wise (possibly several small tensors for representing a single instance mask)
    boxes = boxes.to(torch.device("cpu"))

    results = [
        rasterize_polygons_within_box(mask, box, mask_side_len) for mask, box in zip(masks, boxes)
    ]
    """
    mask: list[list[float]], the polygons for one instance
    box: a tensor of shape (4,)
    """

    if len(results) == 0:
        return torch.empty(0, dtype=torch.uint8, device=device)
    return torch.stack(results, dim=0).to(device=device)


def batch_rasterize_full_image_polygons_within_box(masks, boxes, mask_size):
    """
    Batched version of :func:`rasterize_polygons_within_box`.
    Compared to :func:`batch_rasterize_polygons_within_box`,
    this function uses ROIAlign on full image masks,
    and has less reconstruction error.
    However we observe no difference in accuracy, therefore
    we still use the version :func:`batch_rasterize_polygons_within_box`,
    which is faster and uses less memory.

    Args:
        masks (PolygonMasks): store N masks for an image in polygon format.
        boxes (Tensor): store N boxes corresponding to the masks.
        mask_size (int): the size of the rasterized mask.

    Returns:
        Tensor: A byte tensor of shape (N, mask_size, mask_size), where
            N is the number of predicted boxes for this image.
    """
    device = boxes.device
    boxes = boxes.cpu()
    maxx, maxy = boxes[:, 2:].max(dim=0)[0]
    height, width = maxy + 1, maxx + 1

    bit_masks = []  # #instance x H x W
    for polygons_per_instance in masks:
        rles = mask_utils.frPyObjects(polygons_per_instance, height, width)
        rle = mask_utils.merge(rles)
        bit_mask = mask_utils.decode(rle)
        bit_masks.append(bit_mask)
    bit_masks = torch.from_numpy(np.asarray(bit_masks, dtype="float32"))

    batch_inds = torch.arange(len(boxes)).to(dtype=boxes.dtype)[:, None]
    # See comments in roi_align.py about -0.5
    rois = torch.cat([batch_inds, boxes - 0.5], dim=1)  # Nx5

    bit_masks = bit_masks.to(device=device)
    rois = rois.to(device=device)
    output = (
        ROIAlign((mask_size, mask_size), 1.0, 0).forward(bit_masks[:, None, :, :], rois).squeeze(1)
    )
    output = output >= 0.5
    return output


class PolygonMasks(object):
    """
    This class stores the segmentation masks for all objects in one image, in the form of polygons.
    """

    def __init__(self, polygons):
        """
        Arguments:
            polygons (list[list[list[float]]]): The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                instance, and the third level to the polygon coordinates.
                The third level list should have the format of
                [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
        """
        assert isinstance(polygons, list)

        def process_polygons(polygons_per_instance):
            assert isinstance(polygons_per_instance, list), type(polygons_per_instance)
            # transform the polygon to a tensor
            polygons_per_instance = [
                # use float64 for higher precision, because why not?
                torch.as_tensor(p, dtype=torch.float64)
                for p in polygons_per_instance
            ]
            for polygon in polygons_per_instance:
                assert len(polygon) % 2 == 0 and len(polygon) >= 6
            return polygons_per_instance

        self.polygons = [
            process_polygons(polygons_per_instance) for polygons_per_instance in polygons
        ]  # list[list[Tensor]]

    def to(self, *args, **kwargs):
        return self

    def __getitem__(self, item):
        """
        Support indexing over the instances and return a `PolygonMasks` object.
        `item` can be:
            1. An integer. It will return an object with only one instance.
            2. A slice. It will return an object with the selected instances.
            3. A vector mask of type ByteTensor, whose length is num_instances.
               It will return an object with the instances whose mask is nonzero.
        """
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        else:
            # advanced indexing on a single dimension
            if isinstance(item, torch.Tensor) and item.dtype == torch.uint8:
                assert item.dim() == 1, item.shape
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            selected_polygons = [self.polygons[i] for i in item]
        return PolygonMasks(selected_polygons)

    def __iter__(self):
        return iter(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.polygons))
        return s

    def __len__(self):
        return len(self.polygons)

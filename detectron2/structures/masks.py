import copy
import numpy as np
import pycocotools.mask as mask_utils
import torch

from detectron2.layers.roi_align import ROIAlign

from .boxes import Boxes


def polygons_to_bitmask(polygons, height, width):
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    assert len(polygons) > 0, "COCOAPI does not support empty polygons"
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(np.bool)


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
        Tensor: BoolTensor of shape (mask_size, mask_size)
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
    mask = polygons_to_bitmask(polygons, mask_size, mask_size)
    mask = torch.from_numpy(mask)
    return mask


def batch_crop_and_resize(masks, boxes, mask_size):
    """
    Crop regions in masks using boxes, and produce result images in the size
    of mask_size x mask_size.

    Args:
        masks (PolygonMasks or BitMasks): store N masks for an image in polygon
            format or bitmap format.
        boxes (Tensor): store N boxes corresponding to the masks.
        mask_size (int): the size of the output mask.

    Returns:
        Tensor: A bool tensor of shape (N, mask_size, mask_size), where
            N is the number of predicted boxes for this image.
    """
    if isinstance(masks, PolygonMasks):
        device = boxes.device
        # Put boxes on the CPU, as the polygon representation is not efficient GPU-wise
        # (several small tensors for representing a single instance mask)
        boxes = boxes.to(torch.device("cpu"))

        results = [
            rasterize_polygons_within_box(mask, box, mask_size) for mask, box in zip(masks, boxes)
        ]
        """
        mask: list[list[float]], the polygons for one instance
        box: a tensor of shape (4,)
        """
        if len(results) == 0:
            return torch.empty(0, dtype=torch.bool, device=device)
        return torch.stack(results, dim=0).to(device=device)
    else:
        assert isinstance(masks, BitMasks), type(masks)
        return batch_crop_and_resize_bitmask(masks, boxes, mask_size)


def batch_crop_and_resize_bitmask(masks, boxes, mask_size):
    """
    Use ROIAlign to rasterize boxes in full image BitMasks.
    It has less reconstruction error compared to rasterization with polygons.
    However we observe no difference in accuracy,
    but this one uses more memory to store all the masks.

    Args:
        masks (BitMasks): store N masks for an image in polygon format.
        boxes (Tensor): store N boxes corresponding to the masks.
        mask_size (int): the size of the rasterized mask.

    Returns:
        Tensor: A bool tensor of shape (N, mask_size, mask_size), where
            N is the number of predicted boxes for this image.
    """
    device = masks.tensor.device

    batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
    rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5

    bit_masks = masks.tensor.to(dtype=torch.float32)
    rois = rois.to(device=device)
    output = (
        ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
        .forward(bit_masks[:, None, :, :], rois)
        .squeeze(1)
    )
    output = output >= 0.5
    return output


class BitMasks:
    """
    This class stores the segmentation masks for all objects in one image, in
    the form of bitmaps.

    Attributes:
        tensor: bool Tensor of N,H,W, representing N instances in the image.
    """

    def __init__(self, tensor):
        """
        Args:
            tensor: bool Tensor of N,H,W, representing N instances in the image.
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.bool, device=device)
        assert tensor.dim() == 3, tensor.size()
        self.image_size = tensor.shape[1:]
        self.tensor = tensor

    def to(self, device):
        return BitMasks(self.tensor.to(device))

    def __getitem__(self, item):
        """
        Returns:
            BitMasks: Create a new :class:`BitMasks` by indexing.

        The following usage are allowed:
        1. `new_masks = masks[3]`: return a `BitMasks` which contains only one mask.
        2. `new_masks = masks[2:10]`: return a slice of masks.
        3. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return BitMasks(self.tensor[item].view(1, -1))
        m = self.tensor[item]
        assert m.dim() == 3, "Indexing on BitMasks with {} returns a tensor with shape {}!".format(
            item, m.shape
        )
        return BitMasks(m)

    def __iter__(self):
        yield from self.tensor

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    def __len__(self):
        return self.tensor.shape[0]

    def nonempty(self):
        """
        Find masks that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
        """
        return self.tensor.flatten(1).any(dim=1)

    @staticmethod
    def from_polygon_masks(polygon_masks, height, width):
        """
        Args:
            polygon_masks (list[list[ndarray]] or PolygonMasks)
            height, width (int)
        """
        if isinstance(polygon_masks, PolygonMasks):
            polygon_masks = polygon_masks.numpy()
        masks = [polygons_to_bitmask(p, height, width) for p in polygon_masks]
        return BitMasks(torch.stack([torch.from_numpy(x) for x in masks]))

    def get_bounding_boxes(self):
        # not needed now
        raise NotImplementedError


class PolygonMasks:
    """
    This class stores the segmentation masks for all objects in one image, in the form of polygons.

    Attributes:
        polygons: list[list[Tensor]]. Each Tensor is a float64 vector representing a polygon.
    """

    def __init__(self, polygons):
        """
        Arguments:
            polygons (list[list[Tensor[float]]]): The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                instance, and the third level to the polygon coordinates.
                The third level Tensor should have the format of
                torch.Tensor([x0, y0, x1, y1, ..., xn, yn]) (n >= 3).
        """
        assert isinstance(polygons, list)

        def process_polygons(polygons_per_instance):
            assert isinstance(polygons_per_instance, list), type(polygons_per_instance)
            # transform the polygon to a tensor
            polygons_per_instance = [
                # Use float64 for higher precision, because why not?
                # Always put polygons on CPU (self.to is a no-op) since they
                # are supposed to be small tensors.
                # May need to change this assumption if GPU placement becomes useful
                torch.as_tensor(p, dtype=torch.float64).cpu()
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

    def get_bounding_boxes(self):
        """
        Returns:
            Boxes: tight bounding boxes around polygon masks.
        """
        boxes = torch.zeros(len(self.polygons), 4, dtype=torch.float32)
        for idx, polygons_per_instance in enumerate(self.polygons):
            minxy = torch.as_tensor([float("inf"), float("inf")], dtype=torch.float32)
            maxxy = torch.zeros(2, dtype=torch.float32)
            for polygon in polygons_per_instance:
                coords = polygon.view(-1, 2).to(dtype=torch.float32)
                minxy = torch.min(minxy, torch.min(coords, dim=0).values)
                maxxy = torch.max(maxxy, torch.max(coords, dim=0).values)
            boxes[idx, :2] = minxy
            boxes[idx, 2:] = maxxy
        return Boxes(boxes)

    def nonempty(self):
        """
        Find masks that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
        """
        keep = [1 if len(polygon) > 0 else 0 for polygon in self.polygons]
        return torch.as_tensor(keep, dtype=torch.bool)

    def __getitem__(self, item):
        """
        Support indexing over the instances and return a `PolygonMasks` object.
        `item` can be:
            1. An integer. It will return an object with only one instance.
            2. A slice. It will return an object with the selected instances.
            3. A list[int]. It will return an object with the selected instances,
                correpsonding to the indices in the list.
            4. A vector mask of type BoolTensor, whose length is num_instances.
               It will return an object with the instances whose mask is nonzero.
        """
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        elif isinstance(item, list):
            selected_polygons = [self.polygons[i] for i in item]
        else:
            # advanced indexing on a single dimension
            if isinstance(item, torch.Tensor) and item.dtype == torch.bool:
                assert item.dim() == 1, item.shape
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            selected_polygons = [self.polygons[i] for i in item]
        return PolygonMasks(selected_polygons)

    def __iter__(self):
        """
        Yields:
            list[Tensor]: the polygons for one instance. Each Tensor is a
                float64 vector representing a polygon.
        """
        return iter(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.polygons))
        return s

    def __len__(self):
        return len(self.polygons)

    def numpy(self):
        """
        Returns:
            list[list[ndarray]]: the polygons in numpy ndarray format.
        """
        return [[x.numpy() for x in p] for p in self.polygons]

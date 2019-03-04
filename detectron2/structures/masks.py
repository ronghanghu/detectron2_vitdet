import copy
import pycocotools.mask as mask_utils
import torch

# TODO TO_REMOVE maybe remove crop&resize and use roialign when we deal with masks with fewer quantizations


def rasterize_polygons_within_box(polygons, box, mask_size):
    """
    Rasterize the polygons into a mask image and
    crop the mask content in the given box.
    The copped mask is resized to (mask_size, mask_size).

    Args:
        polygons (list[Tensor[float]]): a list of polygons, which represents an instance.
        box: 4 elements
        mask_size (int):
    """
    # 1. Shift the polygons w.r.t the boxes
    w, h = box[2] - box[0], box[3] - box[1]

    # Check if necessary
    w = max(w, 1)
    h = max(h, 1)

    polygons = copy.deepcopy(polygons)
    for p in polygons:
        p[0::2] = p[0::2] - box[0]
        p[1::2] = p[1::2] - box[1]

    # 2. Rescale the polygons to the new box size
    ratio_h = mask_size / h
    ratio_w = mask_size / w

    if ratio_h == ratio_w:
        for p in polygons:
            p *= ratio_h
    else:
        for p in polygons:
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h

    # 3. Rasterize the polygons with coco api
    rles = mask_utils.frPyObjects([p.numpy() for p in polygons], mask_size, mask_size)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    mask = torch.from_numpy(mask)
    return mask


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
                torch.as_tensor(p, dtype=torch.float32) for p in polygons_per_instance
            ]
            for polygon in polygons_per_instance:
                assert len(polygon) % 2 == 0 and len(polygon) >= 6
            return polygons_per_instance

        self.polygons = [
            process_polygons(polygons_per_instance) for polygons_per_instance in polygons
        ]

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

import torch

from detectron2.layers import cat


class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)
    """

    def __init__(self, tensor, mode: str = "xyxy"):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.
                Each row is either (x1, y1, x2, y2), or (x1, y1, w, h) (see `mode`).
            mode (str): either "xyxy" or "xywh". The meaning of each row in tensor.
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()
        assert mode in ("xyxy", "xywh"), "mode should be 'xyxy' or 'xywh'! Got {}".format(mode)

        self.tensor = tensor
        self.mode = mode

    def clone(self, mode=None):
        """
        Clone the Boxes (and optionally convert to a different mode)

        Args:
            mode (str or None): either "xyxy" or "xywh". If not None, will convert the mode of the Boxes.

        Returns:
            Boxes
        """
        if mode is None:
            mode = self.mode
        assert mode in ("xyxy", "xywh"), "mode should be 'xyxy' or 'xywh'! Got {}".format(mode)
        if mode == self.mode:
            return Boxes(self.tensor.clone(), self.mode)
        # we only have two modes, so no need to check self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            tensor = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            boxes = Boxes(tensor, mode=mode)
        else:
            TO_REMOVE = 1
            tensor = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            boxes = Boxes(tensor, mode=mode)
        return boxes

    def _split_into_xyxy(self):
        """"
        Returns:
            4 vectors: x1, y1, x2, y2
        """
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.tensor.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.tensor.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def to(self, device):
        return Boxes(self.tensor.to(device), self.mode)

    def area(self):
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")
        return area

    def clip(self, image_size):
        """
        Clip (in place) the boxes to the shape of the image.

        Args:
            image_size (h, w)
        """
        assert self.mode == "xyxy"
        h, w = image_size
        TO_REMOVE = 1
        self.tensor[:, 0].clamp_(min=0, max=w - TO_REMOVE)
        self.tensor[:, 1].clamp_(min=0, max=h - TO_REMOVE)
        self.tensor[:, 2].clamp_(min=0, max=w - TO_REMOVE)
        self.tensor[:, 3].clamp_(min=0, max=h - TO_REMOVE)

    def nonempty(self, threshold=0):
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor: a binary vector which represents whether each box is empty (False) or non-empty (True).
        """
        assert self.mode == "xyxy"
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item):
        """
        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:
        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.ByteTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1), self.mode)
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b, self.mode)

    def __len__(self):
        return self.tensor.shape[0]

    def inside_image(self, image_size, boundary_threshold=0):
        """
        Args:
            image_size (h, w):
            boundary_threshold (int): Boxes that extend beyond the image
                boundary by more than boundary_threshold are considered "outside".
                Can be set to negative value to determine more boxes to be "outside".
                Setting it to a very large number will effectively disable this behavior.

        Returns:
            a binary vector, indicating whether each box is inside the image.
        """
        assert self.mode == "xyxy"
        height, width = image_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    @staticmethod
    def cat(boxes_list):
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        assert all(isinstance(box, Boxes) for box in boxes_list)

        mode = boxes_list[0].mode
        assert all(b.mode == mode for b in boxes_list)

        cat_boxes = Boxes(cat([b.tensor for b in boxes_list], dim=0), mode)
        return cat_boxes

    @property
    def device(self):
        return self.tensor.device

    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at at time.
        """
        yield from self.tensor


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1, boxes2):
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
      Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()
    area2 = boxes2.area()

    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

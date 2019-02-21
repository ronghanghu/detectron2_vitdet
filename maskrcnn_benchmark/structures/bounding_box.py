import torch

from maskrcnn_benchmark.layers import cat


# TODO(yuxinwu) reorganize the files


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
        Create a new Boxes by indexing on this Boxes.

        It returns `Boxes(self.tensor[item], self.mode)`.

        Note that the returned Boxes might share storage with this Boxes, subject to Pytorch's indexing semantics.
        """
        b = self.tensor[item]
        if b.dim() == 1:
            b = b.view(1, -1)
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


class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same `__len__` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:
    1. Set/Get a field:
        instances.gt_boxes = Boxes(...)
        print(instances.pred_masks)
        print('gt_masks' in instances)

    2. `len(instance)` returns the number of instances

    3. Indexing: `instance[indices]` will apply the indexing on all the fields
       and returns a new `Instances`.
       Typically, `indices` is a binary vector of length num_instances,
       or a vector of integer indices.
    """

    def __init__(self, image_size, **kwargs):
        """
        Args:
            image_size (h, w): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size  # (image_width, image_height)
        self._fields = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self):
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name, val):
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name):
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name, value):
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name):
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def get(self, name):
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self):
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, device):
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if
                the field has this method.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            ret.set(k, v)
        return ret

    def __getitem__(self, item):
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self):
        for v in self._fields.values():
            return len(v)
        raise NotImplementedError("Empty Instances does not support __len__!")

    @staticmethod
    def cat(instance_lists):
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            if isinstance(values[0], torch.Tensor):
                values = cat(values, dim=0)
            elif isinstance(values[0], Boxes):
                values = Boxes.cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(values[0])))
            ret.set(k, values)
        return ret

    def __str__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_width={}, ".format(self._image_size[0])
        s += "image_height={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join(self._fields.keys()))
        return s

import torch

from detectron2.layers import cat

from .boxes import Boxes


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

    def remove(self, name):
        """
        Remove the field called `name`.
        """
        del self._fields[name]

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

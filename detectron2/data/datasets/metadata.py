import types


class Metadata(types.SimpleNamespace):
    """
    A class that supports simple attribute setter/getter.
    It is intended for storing metadata of a dataset and make it accessible globally.

    Examples:

    .. code-block:: python

        # somewhere when you load the data:
        MetadataCatalog.get("mydataset").class_names = ["person", "dog"]

        # somewhere when you print statistics or visualize:
        class_names = MetadataCatalog.get("mydataset").class_names
    """

    name: str  # the name of the dataset

    def __getattr__(self, key):
        raise AttributeError(
            "Attribute {} does not exist in metadata for {}".format(key, self.name)
        )

    def __setattr__(self, key, val):
        # Ensure that metadata of the same dataset stays consistent
        try:
            oldval = getattr(self, key)
            assert (
                oldval == val
            ), "Setting attribute {} of metadata {} to a different value!\n{} != {}".format(
                key, self.name, oldval, val
            )
        except AttributeError:
            super().__setattr__(key, val)


class MetadataCatalog:
    """
    Metadata catalog provides access to "Metadata" of a given dataset (e.g., "coco", "cityscapes").
    The metadata associated with a certain name is a singleton: once created,
    the metadata will stay alive and will be returned by future calls to `get(name)`.
    """

    _NAME_TO_META = {}

    @staticmethod
    def get(name):
        """
        Args:
            name (str): Note that this is not the name of a dataset split (e.g. coco_2014_train),
                but the name of the dataset (e.g., coco)

        Returns:
            Metadata: The :class:`Metadata` instance associated with this name,
                or create an empty one if none is available.
        """
        assert len(name)
        name = name.lower()
        if name in MetadataCatalog._NAME_TO_META:
            return MetadataCatalog._NAME_TO_META[name]
        else:
            m = MetadataCatalog._NAME_TO_META[name] = Metadata(name=name)
            return m


def _add_predefined_metadata():
    # coco:
    meta = MetadataCatalog.get("coco")
    # fmt: off
    # Mapping from the incontiguous COCO category id to an id in [1, 80]
    meta.json_id_to_contiguous_id = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}  # noqa
    # 80 names for COCO
    meta.class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]  # noqa
    # fmt: on

    # coco_person:
    meta = MetadataCatalog.get("coco_person")
    meta.class_names = ["person"]
    # TODO add COCO keypoint names, or are they needed at all?


# We hard-coded some metadata. This will enable:
# 1. Consistency check when loading the datasets
# 2. Use models on these standard datasets directly without having the dataset annotations
_add_predefined_metadata()

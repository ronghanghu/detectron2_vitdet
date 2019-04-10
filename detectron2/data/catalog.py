import copy
import types


class DatasetCatalog(object):
    """
    A catalog that stores information about the splits of datasets and how to obtain them.

    It contains a mapping from strings
    (which are the names of a dataset split, e.g. "coco_2014_train")
    to a function which parses the dataset and returns the samples in the
    format of `list[dict]`.

    The returned dicts should be in Detectron2 Dataset format (See DATASETS.md for details)
    if used with the data loader functionatilities in `detectron2/detection`.

    The purpose of having this catalog is to make it easy to choose
    different datasets, by just using the strings in the config.
    """

    _REGISTERED_SPLITS = {}

    @staticmethod
    def register(key, func):
        """
        Args:
            key (str): the key that identifies a split of a dataset, e.g. "coco_2014_train".
            func (callable): a callable which takes no arguments and returns a list of dicts.
        """
        DatasetCatalog._REGISTERED_SPLITS[key] = func

    @staticmethod
    def get(key):
        """
        Call the registered function and return its results.

        Args:
            key (str): the key that identifies a split of a dataset, e.g. "coco_2014_train".

        Returns:
            list[dict]: dataset annotations.0
        """
        return DatasetCatalog._REGISTERED_SPLITS[key]()


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
            "Attribute '{}' does not exist in the metadata of '{}'".format(key, self.name)
        )

    def __setattr__(self, key, val):
        # Ensure that metadata of the same name stays consistent
        try:
            oldval = getattr(self, key)
            assert oldval == val, (
                "Attribute '{}' in the metadata of '{}' cannot be set "
                "to a different value!\n{} != {}".format(key, self.name, oldval, val)
            )
        except AttributeError:
            super().__setattr__(key, val)

    def as_dict(self):
        """
        Returns all the metadata as a dict.
        Note that modifications to the returned dict will not reflect on the Metadata object.
        """
        return copy.copy(self.__dict__)

    def set(self, **kwargs):
        """
        Set multiple metadata from kwargs.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)


class MetadataCatalog:
    """
    MetadataCatalog provides access to "Metadata" of
    a given dataset (e.g., "coco", "cityscapes") or dataset split.

    The metadata associated with a certain name is a singleton: once created,
    the metadata will stay alive and will be returned by future calls to `get(name)`.

    It's like global variables, so don't abuse it.
    It's meant for storing knowledge that's constant and shared across the execution
    of the program, e.g.: the class names in COCO.
    """

    _NAME_TO_META = {}

    @staticmethod
    def get(name):
        """
        Args:
            name (str): A string that is either a dataset split (e.g. coco_2014_train),
                or the name of the dataset (e.g., coco)

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

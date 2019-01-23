from abc import ABCMeta, abstractmethod
import torch.nn as nn


class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self):
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str: Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        pass

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return self._size_divisibility

    @property
    def return_features(self):
        """
        List of feature maps names that will be returned by :meth:`forward`.
        """
        return self._return_features

    @property
    def feature_strides(self):
        """
        Dict containing strides (values) of each named feature map (keys) produced
        by the backbone. Example keys (depending on backbone type): "stem", "res2",
        ..., "res5".
        """
        return self._feature_strides

    @property
    def feature_channels(self):
        """
        Dict containing the number of channels (values) in each named feature map
        (keys) produced by the backbone. Example keys (depending on backbone type):
        "stem", "res2", ..., "res5".
        """
        return self._feature_channels

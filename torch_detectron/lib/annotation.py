# from abc import ABCMeta

# class DetectionAnnotation(metaclass=ABCMeta):
class DetectionAnnotation():
    """
    An Abstract Base Class defining methods for working with object detection
    annotations for an image. Because the different datasets have different
    formats for their metadata, we define a common interface here that they can
    inherit from, so that operations such as getting the ground-truth bounding
    boxes are consistent across datasets.
    """

    # @abstractmethod
    def bounding_box():
        """
        Get the ground truth bounding box associated with this annotation.
        """
        raise NotImplementedError()
        # pass

    # @abstractmethod
    def class_label():
        """
        Get the class label associated with this annotation.
        """
        raise NotImplementedError()
        # pass

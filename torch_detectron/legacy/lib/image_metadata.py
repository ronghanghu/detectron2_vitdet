# from abc import ABCMeta, abstractmethod

# class ImageMetadata(metaclass=ABCMeta):
class ImageMetadata():
    """
    Defines methods for interacting with an Image. For now we just say it has a
    way of getting DetectionAnnotation's associated with it.
    """

    # @abstractmethod
    def detection_annotations():
        """
        Get the ground truth annotations associated with this image.
        """
        # pass
        raise NotImplementedError()

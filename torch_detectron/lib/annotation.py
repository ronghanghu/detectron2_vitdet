class DetectionAnnotations():
    """
    An Abstract Base Class defining methods for working with object detection
    annotations for an image. Because the different datasets have different
    formats for their metadata, we define a common interface here that they can
    inherit from, so that operations such as getting the ground-truth bounding
    boxes are consistent across datasets.
    """

    def bounding_boxes():
        """
        Get the ground truth bounding boxes associated with this Image.
        """
        raise NotImplementedError()

    # @abstractmethod
    def class_labels():
        """
        Get the class labels associated with this Image.
        """
        raise NotImplementedError()

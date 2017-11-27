from .image_metadata import ImageMetadata
from .annotation import DetectionAnnotation

# Note: this seems like a sus design right now, will have to iterate
class PascalAnnotation(DetectionAnnotation):

    def __init__(self, bounding_box, class_label):
        self.bbox = bounding_box
        self.label = class_label

    def bounding_box(self):
        return self.bbox

    def class_label(self):
        return self.label

class PascalImage(ImageMetadata):

    def __init__(self, raw_annotation):
        bounding_boxes = raw_annotation.get('boxes', list())
        class_labels = raw_annotation.get('classes', list())
        self.annotations = [PascalAnnotation(a[0], a[1]) for a in
                zip(bounding_boxes, class_labels)]

    def detection_annotations(self):
        return self.annotations

def pascal_target_transform(raw_target):
    # raw_target is a dictionary containing two items -->
    # 1. bounding boxes, keyed by 'boxes'
    # 2. class labels, keyed by 'classes'
    return PascalImage(raw_target)

if __name__ == '__main__':
    img = PascalImage({})
    print(img.detection_annotations())

    

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

# not sure about design, but this could also in theory subclass
# something like 'SegmentationAnnotation'
class COCOAnnotation(DetectionAnnotation):

    def __init__(self, raw_annotation):
        self.raw_annotation = raw_annotation

    def bounding_box(self):
        return self.raw_annotation['bbox']

    def class_label(self):
        return self.raw_annotation['category_id']

class COCOImage(ImageMetadata):
 
    def __init__(self, raw_annotation):
        self.annotations = [COCOAnnotation(ann) for ann in raw_annotation]

    def detection_annotations(self):
        return self.annotations

def coco_target_transform(raw_target):
    # coco bounding box is in the format (x, y, width, height) but
    # we want our format to be (x1, y1, x2, y2)
    # TODO: not yet sure what the standard format is, but we can
    # always move the modification to pascal voc if this is correct
    for ann in raw_target:
        bbox = ann['bbox']
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        ann['bbox'] = bbox

    return COCOImage(raw_target)

if __name__ == '__main__':
    img = PascalImage({})
    print(img.detection_annotations())

    

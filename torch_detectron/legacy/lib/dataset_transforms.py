from .image_metadata import ImageMetadata
from .annotation import DetectionAnnotations
from .data_utils import *

def _pascal_bbox_unpack(obj):
    return [obj['bndbox']['xmin'],
            obj['bndbox']['ymin'],
            obj['bndbox']['xmax'],
            obj['bndbox']['ymax']]

def _pascal_bbox_pack(obj, unpacked):
    obj['bndbox']['xmin'] = unpacked[0]
    obj['bndbox']['ymin'] = unpacked[1]
    obj['bndbox']['xmax'] = unpacked[2]
    obj['bndbox']['ymax'] = unpacked[3]

def _get_pascal_objects(raw_annotation):
    return raw_annotation['annotation'].get('object', list())

class PascalImage(DetectionAnnotations):

    def __init__(self, raw_annotation):
        self.raw_annotation = raw_annotation

    def bounding_boxes(self):
        return [_pascal_bbox_unpack(obj) for obj in
                _get_pascal_objects(self.raw_annotation)]

    def class_labels(self):
        return [obj['name'] for obj in
            self.raw_annotation['annotation']['object']]

# Used to apply a series of transforms to the bounding boxes in a Pascal
# annotation
class PascalObjectTransform(object):

    def __init__(self, funcs):
        self.funcs = funcs

    def __call__(self, raw_annotation):
        for obj in _get_pascal_objects(raw_annotation):
            for f in self.funcs:
                obj = f(unpack)

def _pascal_flip(obj):
    unpacked = _pascal_bbox_unpack(obj)
    image_width = obj['size']['width']
    flipped = bbox_xyxy_hflip(unpacked, image_width)
    _pascal_bbox_pack(obj, flipped)

    # TODO: need to flip size metadata as well

def pascal_target_transform(raw_target, obj_transform):
    raw_target = obj_transform(raw_target)
    return PascalImage(raw_target)

def _coco_bbox_unpack(obj):
    return obj['bbox']

def _coco_bbox_pack(obj, unpacked):
    obj['bbox'] = unpacked

class COCOImage(DetectionAnnotations):
 
    def __init__(self, raw_annotation):
        self.raw_annotation = raw_annotation

    def bounding_boxes(self):
        return [_coco_bbox_unpack(obj) for obj in self.raw_annotation]

    def class_labels(self):
        return [obj['category_id'] for obj in self.raw_annotation]


class COCOObjectTransform(object):

    def __init__(self, funcs, img):
        self.funcs = funcs
        self.img = img

    def __call__(self, raw_annotation):
        for obj in raw_annotation:
            for f in self.funcs:
                unpacked = f(unpacked, img)

def _coco_xywh_to_xyxy(obj):
    # TODO: need to handle segmentation
    bbox = _coco_bbox_unpack(obj)
    modified = bbox_xywh_to_xyxy(bbox)
    _coco_bbox_pack(obj, modified)

def _coco_flip(obj, img):
    # expects in xyxy form?
    bbox = _coco_bbox_unpack(obj)
    flipped = bbox_xyxy_flip(obj, img.width)
    _coco_bbox_pack(obj, flipped)

def coco_joint_transform(img, target, img_transform, obj_transform):
    # coco bounding box is in the format (x, y, width, height) but
    # we want our format to be (x1, y1, x2, y2)

    # TODO: not yet sure what the standard format is, but we can
    # always move the modification to pascal voc if this is correct

    # TODO: Use object transform here
    # TODO: Use img transform here

    return img, COCOImage(target)

if __name__ == '__main__':
    img = PascalImage({})
    print(img.detection_annotations())

    

def bbox_xywh_hflip(box, image_width):
    """Horizonatally flip a bounding box in XYWH format"""

    # In a horizontal flip, the y offsets of the image
    # do not change but the x offset is changed so that
    # it becomes the distance between the end of the
    # bounding box and end of the image

    # assumes (x, y, w, h) format
    # todo: do we makes copies?

    offset = image_width - (box[0] + box[2])
    box[0] = offset
    return box

def bbox_xyxy_hflip(box, image_width):
    """Horizonatally flip a bounding box in XYXY format"""

    box[0] = image_width - 1 - box[0]
    box[2] = image_width - 1 - box[2]
    temp = box[0]
    box[0] = box[2]
    box[2] = temp

    return box

def bbox_xywh_to_xyxy(box):
    """Convert a box in format XYWH to XYXY"""

    box[0] += box[2] - 1
    box[1] += box[3] - 1
    
    return box

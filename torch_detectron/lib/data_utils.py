def bbox_xywh_hflip(box, image_width):
    """Horizonatally flip a bounding box"""

    # In a horizontal flip, the y offsets of the image
    # do not change but the x offset is changed so that
    # it becomes the distance between the end of the
    # bounding box and end of the image

    # assumes (x, y, w, h) format
    # todo: do we makes copies?

    offset = image_width - (box[0] + box[2])
    box[0] = offset
    return box

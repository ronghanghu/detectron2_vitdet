import copy
import numpy as np

from detectron2.structures import Boxes

try:
    import cv2  # noqa
except ImportError:
    # If opencv is not available, everything else should still run
    pass


"""
This module contains some common visualization utilities.
It plots text/boxes/masks/keypoints on an image, with some pre-defined "artistic" style & color.

These functions expect BGR images in (H, W, 3), with values in range [0, 255].
They all return an uint8 image of shape (H, W, 3), and the function may modify the input in-place.
"""


_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)


def draw_text(img, pos, text, font_scale=0.35):
    """
    Draw text on an image.

    Args:
        pos (tuple): x, y; the position of the text
        text (str):
        font_scale (float):
    """
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * text_h)
    back_br = x0 + text_w, y0
    cv2.rectangle(img, back_tl, back_br, _GREEN, -1)
    # Show text.
    text_tl = x0, y0 - int(0.3 * text_h)
    cv2.putText(img, text, text_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img


def draw_boxes(img, boxes, thickness=1):
    """
    Draw boxes on an image.

    Args:
        boxes (Boxes or ndarray): either a :class:`Boxes` instances, or a Nx4 numpy array of xyxy format.
        thickness (int): the thickness of the edges
    """
    img = img.astype(np.uint8)
    if isinstance(boxes, Boxes):
        boxes = boxes.clone("xyxy")
    else:
        assert boxes.ndim == 2, boxes.shape
    for box in boxes:
        (x0, y0, x1, y1) = (int(x + 0.5) for x in box)
        img = cv2.rectangle(img, (x0, y0), (x1, y1), color=_GREEN, thickness=thickness)
    return img


def draw_coco_dict(dataset_dict):
    """
    Draw a COCO-style annotation for an image.

    Args:
        dataset_dict (dict): a COCO-style dict, with "file_name" and "annotations".
    """
    img = cv2.imread(dataset_dict["file_name"])
    annos = dataset_dict["annotations"]
    boxes = np.asarray([k["bbox"] for k in annos])

    # TODO assumes xywh for now.
    # TODO We need to normalize everything to xyxy and limit xywh to be used inside dataset-specific code only

    # Display in largest to smallest order to reduce occlusion
    areas = boxes[:, 2] * boxes[:, 3]
    sorted_inds = np.argsort(-areas)
    sorted_boxes = copy.deepcopy(boxes[sorted_inds])

    sorted_boxes[:, 2:4] += sorted_boxes[:, 0:2]
    img = draw_boxes(img, sorted_boxes)

    for i in sorted_inds:
        anno = annos[i]
        bbox = anno["bbox"]
        # TODO have a centralized way to obtain metadata (e.g., class names) for a given dataset
        cls = str(anno["category_id"])
        img = draw_text(img, (bbox[0], bbox[1] - 2), cls)

    # TODO segmentation / keypoints
    return img

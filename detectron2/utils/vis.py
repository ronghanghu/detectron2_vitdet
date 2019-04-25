import copy
import numpy as np
import pycocotools.mask as mask_util

from detectron2.structures import Boxes, BoxMode

from .colormap import colormap

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
    if x0 + text_w > img.shape[1]:
        x0 = img.shape[1] - text_w
    if y0 - int(1.2 * text_h) < 0:
        y0 = int(1.2 * text_h)
    back_topleft = x0, y0 - int(1.3 * text_h)
    back_bottomright = x0 + text_w, y0
    cv2.rectangle(img, back_topleft, back_bottomright, _GREEN, -1)
    # Show text.
    text_bottomleft = x0, y0 - int(0.2 * text_h)
    cv2.putText(img, text, text_bottomleft, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img


def draw_boxes(img, boxes, thickness=1):
    """
    Draw boxes on an image.

    Args:
        boxes (Boxes or ndarray): either a :class:`Boxes` instances,
            or a Nx4 numpy array of XYXY_ABS format.
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


def draw_polygons(img, polygons, thickness=1):
    """
    Draw polygons on an image.

    Args:
        polygons (list[list[float]] or list[ndarray]): a list of polygons.
            Each polygon is represented by Nx2 numbers: (x0, y0, x1, y1, ...).
    """
    contours = [np.asarray(poly).astype("int32").reshape(-1, 2) for poly in polygons]
    img = cv2.drawContours(img, contours, -1, _WHITE, thickness, cv2.LINE_AA)
    return img


def draw_mask(img, mask, color, alpha=0.4, draw_contours=True):
    """
    Draw (overlay) a mask on an image.

    Args:
        mask (ndarray): an (H, W) array of the same spatial size as the image.
            Nonzero positions in the array are considered part of the mask.
        color: a BGR color
        alpha (float): blending efficient. Smaller values lead to more transparent masks.
        draw_contours (bool): whether to also draw the contours of every
            connected component (object part) in the mask.
    """
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * color

    if draw_contours:
        # opencv func signature has changed between versions
        contours = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2]
        cv2.drawContours(img, contours, -1, _WHITE, 1, cv2.LINE_AA)
    return img.astype(np.uint8)


def draw_keypoints(img, keypoints):
    """
    Args:
        keypoints (ndarray): Nx2 array, each row is an (x, y) coordinate.
    """
    for coord in keypoints:
        cv2.circle(img, tuple(coord), thickness=-1, lineType=cv2.LINE_AA, radius=3, color=_GREEN)
    return img


def draw_coco_dict(dataset_dict, class_names=None):
    """
    Draw the instance annotations for an image.

    Args:
        dataset_dict (dict): a dict in Detectron2 Dataset format. See DATASETS.md
        class_names (list[str] or None): `class_names[cateogory_id]` is the
            name for this category. If not provided, the visualization will
            not contain class names.
            Note that the category ids in Detectron2 Dataset format are 1-based.
            So this list may need a dummy first element.
    """
    img = dataset_dict.get("image", None)
    if img is None:
        img = cv2.imread(dataset_dict["file_name"])
    annos = dataset_dict["annotations"]
    if not len(annos):
        return img
    boxes = np.asarray(
        [BoxMode.convert(k["bbox"], k["bbox_mode"], BoxMode.XYXY_ABS) for k in annos]
    )

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)
    sorted_boxes = copy.deepcopy(boxes[sorted_inds])

    img = draw_boxes(img, sorted_boxes)

    cmap = colormap()

    for num, i in enumerate(sorted_inds):
        anno = annos[i]
        bbox = anno["bbox"]
        assert anno["bbox_mode"] in [
            BoxMode.XYXY_ABS,
            BoxMode.XYWH_ABS,
        ], "Relative coordinates not yet supported in visualization."
        iscrowd = anno.get("iscrowd", 0)
        clsid = anno["category_id"]
        text = class_names[clsid] if class_names is not None else str(clsid)
        if iscrowd:
            text = text + "_crowd"
        img = draw_text(img, (bbox[0], bbox[1] - 2), text)

        segs = anno.get("segmentation", None)
        if segs is not None and not iscrowd:
            segs_color = cmap[num % len(cmap)]
            if isinstance(segs, dict):
                mask = mask_util.decode(segs)
                # We do not draw borders here, so that it will be more obvious to tell from the
                # visualization whether the annotation is in RLE format or polygons.
                img = draw_mask(img, mask, segs_color, draw_contours=False)
            else:
                img = draw_polygons(img, segs)

        kpts = anno.get("keypoints", None)
        if kpts is not None and not iscrowd:
            kpts = np.asarray(kpts).reshape(-1, 3)[:, :2]
            img = draw_keypoints(img, kpts)
    return img


def draw_instance_predictions(img, predictions, class_names=None):
    """
    Draw instance-level prediction results on an image.

    Args:
        predictions (Instances): the output of an instance detection/segmentation
            model. Following fields will be used to draw:
            "pred_boxes", "scores", "pred_classes", "pred_masks" (or "pred_masks_rle").
        class_names (list[str] or None): `class_names[cateogory_id]` is the
            name for this category. If not provided, the visualization will
            not contain class names.
    """

    # TODO do not assume all fields exist, but draw as much as possible,
    # so the function will be more usable for different types of models
    predictions = predictions.to("cpu")
    boxes = predictions.pred_boxes

    # Display in largest to smallest order to reduce occlusion
    areas = boxes.area().numpy()
    sorted_inds = np.argsort(-areas)

    sorted_boxes = copy.deepcopy(boxes.tensor.numpy()[sorted_inds])
    img = draw_boxes(img, sorted_boxes)

    cmap = colormap()

    for num, i in enumerate(sorted_inds):
        bbox = predictions.pred_boxes.tensor[i].numpy()
        cls = predictions.pred_classes[i]
        score = predictions.scores[i]
        text = "{}: {:.2f}".format(class_names[cls] if class_names is not None else cls, score)
        img = draw_text(img, (bbox[0], bbox[1] - 2), text)

        mask = None
        if predictions.has("pred_masks"):
            mask = predictions.pred_masks[i].numpy()
        elif predictions.has("pred_masks_rle"):
            mask_rle = predictions.pred_masks_rle[i]
            mask = mask_util.decode(mask_rle)
        if mask is not None:
            mask_color = cmap[num % len(cmap)]
            img = draw_mask(img, mask, color=mask_color, draw_contours=True)

        if predictions.has("pred_keypoints"):
            keypoints = predictions.pred_keypoints.tensor[i].numpy()
            coords = keypoints.reshape(-1, 3)[:, :2]
            img = draw_keypoints(img, coords)
    return img

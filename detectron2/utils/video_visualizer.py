import numpy as np
import pycocotools.mask as mask_util

from detectron2.utils.visualizer import ColorMode, Visualizer

from .colormap import random_color


class _DetectedInstance:
    """
    Used to store data about detected objects in video frame,
    in order to transfer color to objects in the future frames.

    Attributes:
        label (int):
        bbox (tuple[float]):
        mask_rle (dict):
        color (tuple[float]): RGB colors in range (0, 1)
        ttl (int): time-to-live for the instance. For example, if ttl=2,
            the instance color can be transferred to objects in the next two frames.
    """

    __slots__ = ["label", "bbox", "mask_rle", "color", "ttl"]

    def __init__(self, label, bbox, mask_rle, color, ttl):
        self.label = label
        self.bbox = bbox
        self.mask_rle = mask_rle
        self.color = color
        self.ttl = ttl


class VideoVisualizer:
    def __init__(self, metadata, instance_mode=ColorMode.IMAGE):
        """
        Args:
            metadata (MetadataCatalog): image metadata.
        """
        self.metadata = metadata
        self._old_instances = []
        assert instance_mode in [
            ColorMode.IMAGE,
            ColorMode.IMAGE_BW,
        ], "Other mode not supported yet."
        self._instance_mode = instance_mode

    def draw_instance_predictions(self, frame, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        frame_visualizer = Visualizer(frame, self.metadata)
        num_instances = len(predictions)
        if num_instances == 0:
            return frame_visualizer.output

        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        labels = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = predictions.pred_masks
            # mask IOU is not yet enabled
            # masks_rles = mask_util.encode(np.asarray(masks.permute(1, 2, 0), order="F"))
            # assert len(masks_rles) == num_instances
        else:
            masks = None

        detected = [
            _DetectedInstance(labels[i], boxes[i], mask_rle=None, color=None, ttl=2)
            for i in range(num_instances)
        ]
        colors = self._assign_colors(detected)

        if scores is not None:
            labels = [
                "{}: {:.0f}%".format(self.metadata.class_names[l], s * 100)
                for l, s in zip(labels, scores)
            ]

        if self._instance_mode == ColorMode.IMAGE_BW:
            img_bw = frame.astype("f4").mean(axis=2)
            img_bw = np.stack([img_bw] * 3, axis=2)
            if masks is not None:
                visible = masks.any(dim=0).numpy() > 0
                img_bw[visible] = frame[visible]
            frame_visualizer.output.ax.imshow(img_bw.astype("uint8"))
            alpha = 0.4
        else:
            alpha = 0.5

        frame_visualizer.overlay_instances(
            boxes=None if masks is not None else boxes,  # boxes are a bit distracting
            masks=masks,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )

        return frame_visualizer.output

    def _assign_colors(self, instances):
        """
        Naive tracking heuristics to assign same color to the same instance,
        will update the internal state of tracked instances.

        Returns:
            list[tuple[float]]: list of colors.
        """

        # Compute iou with either boxes or masks:
        is_crowd = np.zeros((len(instances),), dtype=np.bool)
        # if instances[0].mask_rle is not None:
        if False:  # box iou seems good enough
            rles_old = [x.mask_rle for x in self._old_instances]
            rles_new = [x.mask_rle for x in instances]
            ious = mask_util.iou(rles_old, rles_new, is_crowd)
            threshold = 0.7
        else:
            boxes_old = [x.bbox for x in self._old_instances]
            boxes_new = [x.bbox for x in instances]
            ious = mask_util.iou(boxes_old, boxes_new, is_crowd)
            threshold = 0.7
        if len(ious) == 0:
            ious = np.zeros((len(self._old_instances), len(instances)), dtype="float32")

        # Only allow matching instances of the same label:
        for old_idx, old in enumerate(self._old_instances):
            for new_idx, new in enumerate(instances):
                if old.label != new.label:
                    ious[old_idx, new_idx] = 0

        matched_new_per_old = np.asarray(ious).argmax(axis=1)
        max_iou_per_old = np.asarray(ious).max(axis=1)

        # Try to find match for each old instance:
        extra_instances = []
        for idx, inst in enumerate(self._old_instances):
            if max_iou_per_old[idx] > threshold:
                newidx = matched_new_per_old[idx]
                if instances[newidx].color is None:
                    instances[newidx].color = inst.color
                    continue
            # If an old instance does not match any new instances,
            # keep it for the next frame in case it is just missed by the detector
            inst.ttl -= 1
            if inst.ttl > 0:
                extra_instances.append(inst)

        # Assign random color to newly-detected instances:
        for inst in instances:
            if inst.color is None:
                inst.color = random_color(rgb=True, maximum=1)
        self._old_instances = instances[:] + extra_instances
        return [d.color for d in instances]

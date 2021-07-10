# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from PIL import Image

from detectron2.utils.colormap import random_color
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import _OFF_WHITE, ColorMode, Visualizer, _create_text_labels


class YTVISVisualizer(VideoVisualizer):
    def draw_instance_predictions(self, frame, predictions, instance_colors=None):
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
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        obj_ids = predictions.pred_obj_ids if predictions.has("pred_obj_ids") else None
        raw_labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        labels_and_scores = [l.split(" ") for l in raw_labels]
        labels = [
            ls[0] + f" (Object #{obj_id}) " + ls[1]
            for obj_id, ls in zip(obj_ids, labels_and_scores)
        ]

        if predictions.has("pred_masks"):
            masks = predictions.pred_masks
        else:
            masks = None

        if self._instance_mode == ColorMode.IMAGE_BW:
            frame_visualizer.output.img = frame_visualizer._create_grayscale_image(
                (masks.any(dim=0) > 0).numpy() if masks is not None else None
            )
            colors = None
            alpha = 0.3
        else:
            colors = []
            for obj_id in obj_ids:
                obj_id = obj_id.item()
                if obj_id >= 0:
                    color = instance_colors.get(obj_id)
                    if color is None:
                        color = random_color(rgb=True, maximum=1)
                        instance_colors[obj_id] = color
                else:
                    color = _OFF_WHITE
                colors.append(color)
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

    def draw_dataset_dict(self, frame_dict, instance_colors=None):
        """
        Draw annotations/segmentations in Detectron2 Dataset format.

        Args:
            frame_dict (dict): annotation/segmentation data of one frame of a video.
            instance_colors (dict): instances colors to use for visualization

        Returns:
            output (VisImage): image object with visualizations.
        """
        img = np.array(Image.open(frame_dict["file_name"]))
        frame_visualizer = Visualizer(img, self.metadata)
        annos = frame_dict.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None

            from detectron2.structures import BoxMode

            boxes = [
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                if len(x["bbox"]) == 4
                else x["bbox"]
                for x in annos
            ]

            category_ids = [x["category_id"] for x in annos]
            colors = []
            for anno in annos:
                instance_key = (frame_dict["video_id"], anno["instance_id"])
                color = instance_colors.get(instance_key)
                if color is None:
                    color = random_color(rgb=True, maximum=1)
                    instance_colors[instance_key] = color
                colors.append(color)

            names = self.metadata.get("thing_classes", None)
            labels = _create_text_labels(
                category_ids,
                scores=None,
                class_names=names,
                is_crowd=[x.get("iscrowd", 0) for x in annos],
            )
            # Append instance ids to labels for clarity
            labels = [
                l + f" (Instance #{annos[idx]['instance_id']})" for idx, l in enumerate(labels)
            ]
            frame_visualizer.overlay_instances(
                labels=labels, boxes=boxes, masks=masks, keypoints=None, assigned_colors=colors
            )
        return frame_visualizer.output

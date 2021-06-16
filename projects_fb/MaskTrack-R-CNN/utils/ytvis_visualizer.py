# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from PIL import Image

from detectron2.utils.colormap import random_color
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import Visualizer, _create_text_labels


class YTVISVisualizer(VideoVisualizer):
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

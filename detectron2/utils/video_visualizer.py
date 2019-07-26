import numpy as np
from collections import namedtuple
import cv2
import pycocotools.mask as mask_util

from detectron2.structures import PolygonMasks
from detectron2.utils.visualizer import Visualizer

_DISTANCE_BW_POINTS_THRESHOLD = 600
_AREA_DIFFERENCE_THRESHOLD = 3600


"""
Used to store data about different objects in this video frame.

Args:
    center (ndarray[float]): a numpy array containing the x and y coordinates
        that correspond to the mask's center of mass.
    color: color of the polygon. Refer to `matplotlib.colors` for a full list of
        formats that are accepted.
    area (float): the area of the mask.
"""
_ColorScheme = namedtuple("_ColorScheme", "center, color, area")


class VideoVisualizer:
    def __init__(self, metadata):
        """
        Args:
            metadata (MetadataCatalog): image metadata.
        """
        self.metadata = metadata
        self.prev_frame_info = None

    def draw_instance_predictions(self, frame, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            frame (tensor): a tensor of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisualizedImageOutput): image object with visualizations.
            color_metadata (dict): a dictionary, where the keys are the masks
                of the different objects in the current frame and values are
                :class:`_ColorScheme` objects, containing the mask center, color and area.
        """
        frame_visualizer = Visualizer(frame, self.metadata)

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        labels = predictions.pred_classes if predictions.has("pred_classes") else None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        frame_visualizer.overlay_instances(
            boxes=boxes, labels=labels, scores=scores, keypoints=keypoints
        )

        # draw mask
        masks = None
        if predictions.has("pred_masks"):
            masks = predictions.pred_masks
        elif predictions.has("pred_masks_rle"):
            sorted_masks_rle = predictions.pred_masks_rle
            masks = mask_util.decode(sorted_masks_rle)

        # Convert binary masks to vertices of polygon.
        if masks is not None:
            # Display in largest to smallest order to reduce occlusion.
            if boxes is not None:
                areas = boxes.area().numpy()
                sorted_idxs = np.argsort(-areas).tolist()
                if masks is not None:
                    if isinstance(masks, PolygonMasks):
                        masks = masks[sorted_idxs]
                    else:
                        masks = [masks[idx] for idx in sorted_idxs]

            # assign colors to masks
            mask_metadata = self._update_mask_colors_with_prev_frame(masks, frame_visualizer)

            for mask, metadata in zip(masks, mask_metadata):
                # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
                # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
                # Internal contours (holes) are placed in hierarchy-2.
                # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from countours.
                mask_vertices = cv2.findContours(
                    mask.numpy().copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
                )[-2]

                mask_color = metadata.color
                for segment in mask_vertices:
                    segment = np.asarray(segment).reshape(-1, 2)
                    frame_visualizer.draw_polygon(segment, mask_color)
            self.prev_frame_info = mask_metadata

        return frame_visualizer.output

    def _update_mask_colors_with_prev_frame(self, masks, frame_visualizer):
        """
        Looks at previous frame to identify if the same object might have been seen.
        If an object is matched from the previous frame, updates the `color` field of
        the :class:`_ColorScheme` object used to represent different objects in
        the this video frame.

        Args:
            masks (list[Tensor[float]]): the list contains the segmentation masks for all
                objects in one image. Each tensor has a shape of (W, H), where W is the image
                width and H is the image height.
            frame_visualizer (Visualizer): a :class:`Visualizer` object that visualizes the
                current frame.

        Returns:
            mask_metadata (list[_ColorScheme]): a list of :class:`_ColorScheme` objects that
                store metadata related to each mask.
        """
        if self.prev_frame_info is None:
            mask_metadata = []
            for mask in masks:
                # find center of mass for each instance.
                center, area = self._get_center_and_area_of_mask(mask)
                random_color = frame_visualizer._get_color()
                curr_obj = _ColorScheme(center=center, color=random_color, area=area)
                mask_metadata.append(curr_obj)

        else:
            mask_metadata = [None for _ in range(len(masks))]
            matches = []
            for curr_mask_idx, mask in enumerate(masks):
                center, area = self._get_center_and_area_of_mask(mask)
                # get distances to all objects in previous frame.
                for prev_mask_idx, prev_obj in enumerate(self.prev_frame_info):
                    dist = np.linalg.norm(center - prev_obj.center)
                    if (
                        dist < _DISTANCE_BW_POINTS_THRESHOLD
                        and abs(prev_obj.area - area) < _AREA_DIFFERENCE_THRESHOLD
                    ):
                        matches.append([prev_mask_idx, curr_mask_idx, dist])

            # sort by distance
            matches.sort(key=lambda x: x[2])

            # match objects and share color.
            prev_frame_matched_idx = set()
            curr_frame_matched_idx = set()
            for match in matches:
                prev_mask_idx, curr_mask_idx, _ = match

                # share color
                if (
                    prev_mask_idx not in prev_frame_matched_idx
                    and curr_mask_idx not in curr_frame_matched_idx
                ):
                    # update matched indices
                    prev_frame_matched_idx.add(prev_mask_idx)
                    curr_frame_matched_idx.add(curr_mask_idx)

                    # create current mask object
                    prev_obj = self.prev_frame_info[prev_mask_idx]
                    mask = masks[curr_mask_idx]
                    center, area = self._get_center_and_area_of_mask(mask)
                    color = prev_obj.color
                    curr_obj = _ColorScheme(center=center, color=color, area=area)

                    mask_metadata[curr_mask_idx] = curr_obj

            # assign color, center and area values to remaining masks
            for mask_idx, _ in enumerate(masks):
                if not mask_metadata[mask_idx]:
                    mask = masks[mask_idx]
                    center, area = self._get_center_and_area_of_mask(mask)
                    color = frame_visualizer._get_color()
                    curr_obj = _ColorScheme(center=center, color=color, area=area)
                    mask_metadata[mask_idx] = curr_obj

        return mask_metadata

    def _get_center_and_area_of_mask(self, mask):
        """
        Args:
            mask (Tensor[float]): tensor of shape (W, H), where W is the image width
                and H is the image height.

        Returns:
            center (ndarray[float]): a numpy array containing the x and y coordinates
                that correspond to the mask's center of mass.
            area (float): the area of the mask.
        """
        _, _, stats, centroids = cv2.connectedComponentsWithStats(mask.numpy(), 8)
        area = float((stats[1:, -1])[0])
        largest_component_id = np.argmax(area) + 1
        center = centroids[largest_component_id]
        return center, area

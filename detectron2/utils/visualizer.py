import numpy as np
import random
from enum import Enum, unique
import cv2
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from detectron2.structures import PolygonMasks

from .colormap import colormap

_OFF_WHITE = [255, 255, 240]
_BLACK = [0, 0, 0]
_RED = [255, 0, 0]

_KEYPOINT_THRESHOLD = 0.01


@unique
class ColoringMode(Enum):
    """
    Enum of different coloring modes to use during visualizations.

    IMAGE_FOCUSED: Visualizations with overlay. Each region picks a random color (or with some
        rules) from the palette. Colors picked need to have high contrast.
    SEGMENTATION_FOCUSED: Visualizations without overlay. Instances of the same category have
        similar colors.
    """

    IMAGE_FOCUSED = 0
    SEGMENTATION_FOCUSED = 1


class VisualizedImageOutput:
    def __init__(self, img, title="", dpi=75):
        """
        Args:
            img (ndarray): a numpy representation of the image. It has a shape of (H, W, 3), where
                H is the image height, W is the image width and 3 corresponds to the image's RGB
                color channels.
            title (str, optional): image title
            dpi (int, optional): the resolution in dots per inch.
        """
        self.fig, self.ax = self._setup_figure(img, title, dpi)

    def _setup_figure(self, img, title, dpi):
        """
        Args:
            Refer to :meth:`__init__()` in :class:`VisualizedImageOutput`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        img = img.clamp(min=0, max=255).int()
        fig = plt.figure(frameon=False)
        fig.set_size_inches(img.shape[1] / dpi, img.shape[0] / dpi)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_title(title)
        ax.axis("off")
        fig.add_axes(ax)
        ax.imshow(img)
        return fig, ax

    def save_output_file(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)
        plt.close()

    def get_image(self):
        """
        Returns:
            visualized_image (ndarray): a numpy array representation of the visualized image. This
                image has a shape of (H, W, 3), where H is the image height, W is the image width,
                and 3 corresponds to the RGB channels of the image. Each element of the image array
                is of uint8 type.
        """
        canvas = self.fig.canvas
        canvas.draw()
        buffer = np.fromstring(canvas.tostring_rgb(), dtype="uint8")
        num_cols, num_rows = canvas.get_width_height()
        visualized_image = buffer.reshape(num_rows, num_cols, 3)
        return visualized_image


class Visualizer:
    def __init__(self, img_rgb, metadata):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (MetadataCatalog): image metadata.
        """
        self.img = img_rgb
        self.metadata = metadata
        self.output = VisualizedImageOutput(self.img)
        self.cpu_device = torch.device("cpu")

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisualizedImageOutput): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        labels = predictions.pred_classes if predictions.has("pred_classes") else None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        masks, all_mask_vertices = None, None
        if predictions.has("pred_masks"):
            masks = predictions.pred_masks
        elif predictions.has("pred_masks_rle"):
            sorted_masks_rle = predictions.pred_masks_rle
            masks = mask_util.decode(sorted_masks_rle)
        # Convert binary masks to vertices of polygon.
        if masks is not None:
            all_mask_vertices = []
            for mask in masks:
                # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
                # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
                # Internal contours (holes) are placed in hierarchy-2.
                # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from countours.
                mask_vertices = cv2.findContours(
                    mask.numpy().copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
                )[-2]
                all_mask_vertices.append(mask_vertices)

        self.overlay_instances(
            masks=all_mask_vertices, boxes=boxes, labels=labels, scores=scores, keypoints=keypoints
        )
        return self.output

    def draw_sem_seg_predictions(
        self, predictions, area_limit=None, coloring_mode=ColoringMode.SEGMENTATION_FOCUSED
    ):
        """
        Draw stuff prediction results on an image.

        Args:
            predictions (Tensor): the output of a semantic/panotic segmentation model. The tensor of
                shape (H, W).
            area_limit (int): segmenta with less than `area_limit` are not drawn.
            coloring_mode (ColoringMode): mode to use while picking colors for masks.

        Returns:
            output (VisualizedImageOutput): image object with visualizations.
        """
        labels, areas = np.unique(predictions, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels, areas = labels[sorted_idxs], areas[sorted_idxs]
        for label, area in zip(labels, areas):
            # do not draw segments that are too small
            if area_limit and area < area_limit:
                continue

            # draw masks
            if (
                coloring_mode == ColoringMode.SEGMENTATION_FOCUSED
                and self.metadata.stuff_class_names
            ):
                mask_color = self._get_color(class_name=self.metadata.stuff_class_names[label])
                edge_color = [x / 255 for x in _OFF_WHITE]
                alpha = 0.9
            else:
                mask_color = self._get_color()
                edge_color = None
                alpha = 0.5

            binary_mask = (predictions == label).numpy().astype(np.uint8)
            self.draw_binary_mask(binary_mask, color=mask_color, edge_color=edge_color, alpha=alpha)

            # draw text in the center of object
            if self.metadata.stuff_class_names:
                _, _, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
                largest_component_id = np.argmax(stats[1:, -1]) + 1
                center = centroids[largest_component_id]
                self.draw_text(self.metadata.stuff_class_names[label], center)

        return self.output

    def draw_panoptic_seg_predictions(self, panoptic_seg, segments_info, area_limit=None):
        """
        Draw panoptic prediction results on an image.

        Args:
            panoptic_seg (Tensor): of shape (height, width) where the values are ids for each
                segment.
            segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                Each dict contains keys "id", "category_id", "isthing".
            area_limit (int): segments with less than `area_limit` are not drawn.

        Returns:
            output (VisualizedImageOutput): image object with visualizations.
        """
        segment_ids, areas = torch.unique(panoptic_seg, sorted=True, return_counts=True)
        panoptic_seg = panoptic_seg.to(self.cpu_device)
        segment_ids = segment_ids.to(self.cpu_device)
        # Ignore first segment id (i.e. when id is 0), since that is used to indicate
        # that the pixel has no instance or semantic assignment.
        segment_ids = segment_ids[1:]
        alpha = 0.98  # high opacity

        # draw mask for all semantic segments first i.e. "stuff"
        for segment_id, area, segment_info in zip(segment_ids, areas, segments_info):
            if not segment_info["isthing"]:
                # do not draw segments that are too small
                if area_limit and area < area_limit:
                    continue

                # draw mask
                binary_mask = (panoptic_seg == segment_id).numpy().astype(np.uint8)
                category_idx = segment_info["category_id"]
                mask_color = self._get_color(class_name=self.metadata.stuff_class_names[segment_id])
                self.draw_binary_mask(binary_mask, color=mask_color, alpha=alpha)

                # write label at the object's center of mass
                label = self.metadata.stuff_class_names[category_idx]
                _, _, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
                largest_component_id = np.argmax(stats[1:, -1]) + 1
                center = centroids[largest_component_id]
                self.draw_text(label, center)

        # draw mask for all instances second
        for segment_id, segment_info in zip(segment_ids, segments_info):
            if segment_info["isthing"]:
                # draw mask
                category_idx = segment_info["category_id"]
                binary_mask = (panoptic_seg == segment_id).numpy().astype(np.uint8)
                mask_color = self._get_color()
                self.draw_binary_mask(binary_mask, color=mask_color, alpha=alpha)

                # write label at the object's center of mass
                label = self.metadata.class_names[category_idx]
                _, _, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
                largest_component_id = np.argmax(stats[1:, -1]) + 1
                center = centroids[largest_component_id]
                self.draw_text(label, center)

        return self.output

    def overlay_instances(self, masks=None, boxes=None, labels=None, scores=None, keypoints=None):
        """
        Args:
            masks (PolygonMasks i.e. list[list[Tensor[float]]] or list[list[ndarray]]):
                this contains the segmentation masks for all objects in one image. The
                first level of the list corresponds to individual instances. The second
                level to all the polygon that compose the instance, and the third level
                to the polygon coordinates. The third level is either a Tensor or a numpy
                array that should have the format of [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
            boxes (Boxes or ndarray): either a :class:`Boxes` or a Nx4 numpy array
                of XYXY_ABS format for the N objects in a single image.
            labels (Tensor): a tensor of size N, where N is the number of objects in the
                image. Each element in the tensor is the class label of the corresponding
                object in the image. Note that the class labels are integers that are
                in [0, #total classes) range.
            scores (Tensor): a tensor of size N, where N is the number of objects in the image.
                Each element in the tensor is a float in [0.0, 1.0] range, representing the
                confidence of the object's class prediction.
            keypoints (Tensor): a tensor of shape (N, K, 3), where the N is the number of instances
                and K is the number of keypoints. The last dimension corresponds to
                (x, y, probability).

        Returns:
            output (VisualizedImageOutput): image object with visualizations.
        """
        if labels is not None and boxes is None:
            raise ValueError("Cannot overlay labels when there are no boxes.")

        # Display in largest to smallest order to reduce occlusion.
        if boxes is not None:
            areas = boxes.area().numpy()
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlaid instances in descending order.
            boxes = boxes[sorted_idxs]
            labels = labels[sorted_idxs] if labels is not None else labels
            scores = scores[sorted_idxs] if scores is not None else scores
            keypoints = keypoints[sorted_idxs] if keypoints is not None else keypoints
            if masks is not None:
                if isinstance(masks, PolygonMasks):
                    masks = masks[sorted_idxs]
                else:
                    masks = [masks[idx] for idx in sorted_idxs]

        if boxes is not None:
            for idx, box in enumerate(boxes):
                x0, y0, x1, y1 = box
                # draw boxes
                self.draw_box(box)

                # draw text
                if labels is not None:
                    if scores is not None:
                        score = scores[idx]
                        text = "{}: {:.2f}".format(self.metadata.class_names[labels[idx]], score)
                    else:
                        text = self.metadata.class_names[labels[idx]]
                    self.draw_text(text, (x0, y0))

        if masks is not None:
            for mask in masks:
                mask_color = self._get_color()
                for segment in mask:
                    segment = np.asarray(segment).reshape(-1, 2)
                    self.draw_polygon(segment, mask_color)

        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output

    def draw_text(self, text, position, font_size=15, color="g"):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int): font of the text.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.

        Returns:
            output (VisualizedImageOutput): image object with text drawn.
        """
        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            fontsize=font_size,
            family="serif",
            bbox={"facecolor": "none", "alpha": 0.4, "pad": 0, "edgecolor": "none"},
            color=color,
            zorder=10,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="--"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisualizedImageOutput): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0
        self.output.ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=2.5,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def draw_circle(self, circle_coord, color, radius=5):
        """
        Args:
            circle_coord (list(int) or tuple(int)): contains the x and y coordinates
                of the center of the circle.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            radius (int): radius of the circle.

        Returns:
            output (VisualizedImageOutput): image object with box drawn.
        """
        x, y = circle_coord
        self.output.ax.add_patch(plt.Circle(circle_coord, radius=radius, color=color))
        return self.output

    def draw_line(self, x_data, y_data, color):
        """
        Args:
            x_data (list[int]): a list containing x values of all the points being drawn.
                Length of list should match the length of y_data.
            y_data (list[int]): a list containing y values of all the points being drawn.
                Length of list should match the length of x_data.
            color: color of the line. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.

        Returns:
            output (VisualizedImageOutput): image object with line drawn.
        """
        self.output.ax.add_line(Line2D(x_data, y_data, color=color))
        return self.output

    def draw_binary_mask(self, binary_mask, color, edge_color=None, alpha=0.5):
        """
        Args:
            binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
                W is the image width. Each value in the array is either a 0 or 1 value of uint8
                type.
            color: color of the mask. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisualizedImageOutput): image object with mask drawn.
        """
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from countours.
        mask_vertices = cv2.findContours(binary_mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[
            -2
        ]
        for segment in mask_vertices:
            segment = segment.reshape(-1, 2)
            self.draw_polygon(segment, color=color, edge_color=edge_color, alpha=alpha)
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisualizedImageOutput): image object with polygon drawn.
        """
        edge_color = color if edge_color is None else edge_color
        polygon = Polygon(
            segment, fill=True, facecolor=color, edgecolor=edge_color, linewidth=3, alpha=alpha
        )
        self.output.ax.add_patch(polygon)
        return self.output

    def draw_and_connect_keypoints(self, keypoints):
        """
        Draws keypoints of an instance and follows the rules for keypoint connections
        to draw lines between appropriate keypoints. This follows color heuristics for
        line color.

        Args:
            keypoints (Tensor): a tensor of shape (K, 3), where K is the number of keypoints
                and the last dimension corresponds to (x, y, probability).

        Returns:
            output (VisualizedImageOutput): image object with visualizations.
        """
        detected_keypoints_and_locations = {}
        for idx, keypoint in enumerate(keypoints):
            # draw keypoint
            x, y, prob = keypoint
            if prob > _KEYPOINT_THRESHOLD:
                self.draw_circle((x, y), color=_BLACK)
                keypoint_name = self.metadata.keypoint_names[idx]
                detected_keypoints_and_locations[keypoint_name] = (x, y)

        for kp_pairs, color in self.metadata.keypoint_connection_rules.items():
            kp0, kp1 = kp_pairs
            keypoints_present = detected_keypoints_and_locations.keys()
            if kp0 in keypoints_present and kp1 in keypoints_present:
                x0, y0 = detected_keypoints_and_locations[kp0]
                x1, y1 = detected_keypoints_and_locations[kp1]
                color = [x / 255 for x in color]
                self.draw_line([x0, x1], [y0, y1], color=color)

        # draw lines to mid-shoulder and mid-hip
        # Note that this strategy is specific to COCO person keypoints.
        # TODO: Refactor this when visualizer is extended to other datasets.
        nose_x, nose_y, nose_prob = keypoints[self.metadata.keypoint_names.index("nose")]
        ls_x, ls_y, ls_prob = keypoints[self.metadata.keypoint_names.index("left_shoulder")]
        rs_x, rs_y, rs_prob = keypoints[self.metadata.keypoint_names.index("right_shoulder")]
        if ls_prob > _KEYPOINT_THRESHOLD and rs_prob > _KEYPOINT_THRESHOLD:
            color = [x / 255 for x in _RED]
            # draw line from nose to mid-shoulder
            mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
            if nose_prob > _KEYPOINT_THRESHOLD:
                self.draw_line([nose_x, mid_shoulder_x], [nose_y, mid_shoulder_y], color=color)

            # draw line from mid-shoulder to mid-hip
            lh_x, lh_y, lh_prob = keypoints[self.metadata.keypoint_names.index("left_hip")]
            rh_x, rh_y, rh_prob = keypoints[self.metadata.keypoint_names.index("right_hip")]
            if lh_prob > _KEYPOINT_THRESHOLD and rh_prob > _KEYPOINT_THRESHOLD:
                mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
                self.draw_line(
                    [mid_hip_x, mid_shoulder_x], [mid_hip_y, mid_shoulder_y], color=color
                )
        return self.output

    def _get_color(self, class_name=None, idx=None):
        """
        Gives the color corresponding to the class name. If not specified, gives the color
        corresponding to the index. Otherwise, picks a random color from the colormap.

        Args:
            class_name (str, optional): the class of the object. This will be used to
                pick the color assigned to this class.
            idx (int, optional): the index used to pick a color from the colormap list.

        Returns:
            color (list[double]): a list of 3 elements, containing the RGB values of the color
                picked. The values in the list are in the range [0.0, 1.0].
        """
        # pick pre-defined color based on class label.
        if class_name:
            for category in self.metadata.categories:
                if category["name"] == class_name:
                    return np.array(category["color"]).astype(np.float32) / 255
        color_list = colormap(rgb=True) / 255.0
        if idx is None:
            idx = random.randint(0, len(color_list) - 1)
        return color_list[idx % len(color_list)]

    def get_output(self):
        """
        Returns:
            output (VisualizedImageOutput): the image output containing the visualizations added
                to the image.
        """
        return self.output

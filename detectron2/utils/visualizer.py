import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from matplotlib.patches import Polygon

from detectron2.structures import PolygonMasks

from .colormap import colormap


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

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        labels = predictions.pred_classes if predictions.has("pred_classes") else None

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

        self.overlay_instances(masks=all_mask_vertices, boxes=boxes, labels=labels, scores=scores)
        return self.output

    # TODO: Choose color based on heuristics.
    def overlay_instances(self, masks=None, boxes=None, labels=None, scores=None):
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

        Returns:
            output (VisualizedImageOutput): image object with visualizations.
        """
        if labels is not None and boxes is None:
            raise ValueError("Cannot overlay labels when there are no boxes.")

        # Display in largest to smallest order to reduce occlusion.
        if boxes is not None:
            areas = boxes.area().numpy()
            sorted_inds = np.argsort(-areas).tolist()
            # Re-order overlaid instances in descending order.
            boxes = boxes[sorted_inds]
            labels = labels[sorted_inds] if labels is not None else labels
            scores = scores[sorted_inds] if scores is not None else scores
            if masks is not None:
                if isinstance(masks, PolygonMasks):
                    masks = masks[sorted_inds]
                else:
                    masks = [masks[idx] for idx in sorted_inds]

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

    def draw_box(self, boxCoord, alpha=0.5, edgecolor="g", linestyle="--"):
        """
        Args:
            boxCoord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edgecolor: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            linestyle (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisualizedImageOutput): image object with box drawn.
        """
        x0, y0, x1, y1 = boxCoord
        width = x1 - x0
        height = y1 - y0
        self.output.ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edgecolor,
                linewidth=2.5,
                alpha=alpha,
                linestyle=linestyle,
            )
        )
        return self.output

    def draw_polygon(self, segment, color, alpha=0.5):
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisualizedImageOutput): image object with polygon drawn.
        """
        polygon = Polygon(
            segment, fill=True, facecolor=color, edgecolor=color, linewidth=3, alpha=alpha
        )
        self.output.ax.add_patch(polygon)
        return self.output

    def _get_color(self, idx=None):
        """
        Picks a random color from the colormap, unless index specified.

        Args:
            idx (int, optional): the index used to pick a color from the colormap list.

        Returns:
            color (list[double]): a list of 3 elements, containing the RGB values of the color
                picked. The values in the list are in the range [0.0, 1.0].
        """
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

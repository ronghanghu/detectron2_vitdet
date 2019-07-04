import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .colormap import colormap


class VisualizedImageOutput:
    def __init__(self, img, title="", dpi=75):
        self.fig, self.ax = self._setup_figure(img)

    def _setup_figure(self, img, title="", dpi=75):
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
        self.fig.savefig(filepath)
        plt.close()


class Visualizer:
    # TODO: Support other useful options
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

    # TODO: Choose color based on heuristics.
    def overlay_instances(self, masks=None, boxes=None, labels=None):
        """
        Args:
            masks (PolygonMasks i.e. list[list[list[float]]]): this class stores the
                segmentation masks for all objects in one image. The first level of the
                list corresponds to individual instances. The second level to all the
                polygon that compose the instance, and the third level to the
                polygon coordinates. The third level list should have the format of
                [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
            boxes (Boxes or ndarray): either a :class:`Boxes` or a Nx4 numpy array
                of XYXY_ABS format for the N objects in a single image.
            labels (Tensor): a tensor of size N, where N is the number of objects in the
                image. Each element in the tensor is the class label of the corresponding
                object in the image. Note that the class labels are integers that are
                in [0, #total classes) range.

        Returns:
            output (VisualizedImageOutput): image object with visualizations.
        """
        if labels is not None and boxes is None:
            raise ValueError("Cannot overlay labels when there are no boxes.")

        if boxes is not None:
            for idx, box in enumerate(boxes):
                x0, y0, x1, y1 = box
                # draw boxes
                self.draw_box(box)

                # draw text
                if labels is not None:
                    label = labels[idx]
                    text = self.metadata.class_names[label.item()]
                    self.draw_text(text, (x0, y0))

        if masks is not None:
            for mask in masks:
                mask_color = self._get_color()
                for segment in mask:
                    segment = segment.numpy().reshape(-1, 2)
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
        color_list = colormap(rgb=True) / 255
        if idx is None:
            idx = random.randint(0, len(color_list) - 1)
        return color_list[idx % len(color_list)]

    def get_output(self):
        return self.output

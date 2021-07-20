import numpy as np
from typing import List, Optional
from fvcore.transforms.transform import Transform

from detectron2.data.transforms import Augmentation, ResizeTransform


class MosaicTransform(Transform):
    """
    This transform merges together 4 images as described in https://arxiv.org/abs/2004.10934. In
    particular, the 4 images and their bounding boxes are resized and merged on a grid.
    """

    def __init__(
        self,
        output_width: int,
        output_height: int,
        div_x: int,
        div_y: int,
        input_widths: List[int],
        input_heights: List[int],
        interp: Optional[int] = None,
    ) -> None:
        """
        Args:
            output_width, output_height (int): shape of output image
            div_x, div_y (int): dividing locations on the x and y axis separating mosaic inputs
            input_widths (list[int]): list of input image widths
            input_heights (list[int]): list of input image heights
            interp (Optional[int]): PIL interp method for resizing.
        """
        super().__init__()
        assert len(input_widths) == len(input_heights) == 4
        self.resize_transforms = [
            ResizeTransform(input_heights[0], input_widths[0], div_y, div_x, interp=interp),
            ResizeTransform(
                input_heights[1], input_widths[1], div_y, output_width - div_x, interp=interp
            ),
            ResizeTransform(
                input_heights[2], input_widths[2], output_height - div_y, div_x, interp=interp
            ),
            ResizeTransform(
                input_heights[3],
                input_widths[3],
                output_height - div_y,
                output_width - div_x,
                interp=interp,
            ),
        ]
        del input_widths, input_heights
        self._set_attributes(locals())

    def apply_image(
        self, img1: np.ndarray, img2: np.ndarray, img3: np.ndarray, img4: np.ndarray
    ) -> np.ndarray:
        """
        Apply the transform on images. img1 is upper left, img2 is upper right, img3 is lower left,
        and img4 is lower right.

        Args:
            img1, img2, img3, img4 (ndarray): each array is of shape NxH_ixW_ixC, or H_ixW_ixC or
            H_ixW_i. The array can be
                of type uint8 in range [0, 255], or floating point in range [0, 1] or [0, 255].

        Returns:
            ndarray: image after applying the mosaic transformation.
        """
        if not len(img1.shape) == len(img2.shape) == len(img3.shape) == len(img4.shape):
            raise ValueError("Image shapes should be the same length.")
        if not img1.dtype == img2.dtype == img3.dtype == img4.dtype:
            raise ValueError("Image datatypes should be the same.")

        if len(img1.shape) == 2:
            output_image = np.zeros([self.output_height, self.output_width], dtype=img1.dtype)
            output_image[: self.div_y, : self.div_x] = self.resize_transforms[0].apply_image(img1)
            output_image[: self.div_y, self.div_x :] = self.resize_transforms[1].apply_image(img2)
            output_image[self.div_y :, : self.div_x] = self.resize_transforms[2].apply_image(img3)
            output_image[self.div_y :, self.div_x :] = self.resize_transforms[3].apply_image(img4)
        elif len(img1.shape) == 3:
            if not img1.shape[2] == img2.shape[2] == img3.shape[2] == img4.shape[2]:
                raise ValueError("Image Channels should be the same.")

            output_image = np.zeros(
                [self.output_height, self.output_width, img1.shape[2]], dtype=img1.dtype
            )

            output_image[: self.div_y, : self.div_x] = self.resize_transforms[0].apply_image(img1)
            output_image[: self.div_y, self.div_x :] = self.resize_transforms[1].apply_image(img2)
            output_image[self.div_y :, : self.div_x] = self.resize_transforms[2].apply_image(img3)
            output_image[self.div_y :, self.div_x :] = self.resize_transforms[3].apply_image(img4)
        else:
            if not img1.shape[3] == img2.shape[3] == img3.shape[3] == img4.shape[3]:
                raise ValueError("Image Channels should be the same.")
            if not img1.shape[0] == img2.shape[0] == img3.shape[0] == img4.shape[0]:
                raise ValueError("Image Batch Size Should be the Same.")

            output_image = np.zeros(
                [img1.shape[0], self.output_height, self.output_width, img1.shape[2]],
                dtype=img1.dtype,
            )

            output_image[:, : self.div_y, : self.div_x] = self.resize_transforms[0].apply_image(
                img1
            )
            output_image[:, : self.div_y, self.div_x :] = self.resize_transforms[1].apply_image(
                img2
            )
            output_image[:, self.div_y :, : self.div_x] = self.resize_transforms[2].apply_image(
                img3
            )
            output_image[:, self.div_y :, self.div_x :] = self.resize_transforms[3].apply_image(
                img4
            )

        return output_image

    def apply_coords(
        self, coords1: np.ndarray, coords2: np.ndarray, coords3: np.ndarray, coords4: np.ndarray
    ) -> np.ndarray:
        """
        Apply the transform on coordinates.

        Args:
            coords1, coords2, coords3, coords4 (ndarray): each is a floating point array of shape
            N_ix2. The rows represent (x, y) values

        Returns:
            ndarray: all coordinates after applying the mosaic transformation.
        """
        coords1 = self.resize_transforms[0].apply_coords(coords1)
        coords2 = self.resize_transforms[1].apply_coords(coords2)
        coords3 = self.resize_transforms[2].apply_coords(coords3)
        coords4 = self.resize_transforms[3].apply_coords(coords4)

        # coords1 in the upper left (no change)
        # coords2 in the upper right (update x coordinates)
        # coords3 in the lower left (update y coordinates)
        # coords4 in the lower right (update both x and y coordinates)
        coords2[:, 0] += self.div_x
        coords3[:, 1] += self.div_y
        coords4[:, 0] += self.div_x
        coords4[:, 1] += self.div_y

        return np.concatenate([coords1, coords2, coords3, coords4], axis=0)

    def apply_segmentation(
        self,
        segmentation1: np.ndarray,
        segmentation2: np.ndarray,
        segmentation3: np.ndarray,
        segmentation4: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the transform on a full-image segmentation.
        Args:
            segmentation1, segmentation2, segmentation3, segmentation3 (ndarray): each of shape
            H_ixW_i. The array should have integer
            or bool dtype.
        Returns:
            ndarray: segmentation after applying the mosaic transformation.
        """
        return self.apply_image(segmentation1, segmentation2, segmentation3, segmentation4)

    def apply_box(
        self, box1: np.ndarray, box2: np.ndarray, box3: np.ndarray, box4: np.ndarray
    ) -> np.ndarray:
        """
        Apply the transform on a axis-aligned boxes. Will transform the corner points and use
        their minimum/maximum to
        Args:
            box1, box2, box3, bo4 (ndarray): each is N_ix4 floating point array of XYXY format
            in absolute coordinates.
        Returns:
            ndarray: list of all boxes after applying mosaic transformation
        """
        box1 = self.resize_transforms[0].apply_box(box1)
        box2 = self.resize_transforms[1].apply_box(box2)
        box3 = self.resize_transforms[2].apply_box(box3)
        box4 = self.resize_transforms[3].apply_box(box4)

        # box1 in the upper left (no change)
        # box2 in the upper right (update x coordinates)
        # box3 in the lower left (update y coordinates)
        # box4 in the lower right (update both x and y coordinates)
        box2[:, [0, 2]] += self.div_x
        box3[:, [1, 3]] += self.div_y
        box4[:, [0, 2]] += self.div_x
        box4[:, [1, 3]] += self.div_y

        return np.concatenate([box1, box2, box3, box4], axis=0)

    def apply_polygons(
        self,
        polygons1: List[np.ndarray],
        polygons2: List[np.ndarray],
        polygons3: List[np.ndarray],
        polygons4: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Apply the transform on a list of polygons, each represented by a N_ix2 array.
        Args:
            polygon1, polygon2, polygon3, polygon4 (list[ndarray]): each is a Nx2 floating
            point array of (x, y) format in absolute coordinates.
        Returns:
            list[ndarray]: all polygons after apply the transformation.
        """
        polygons = []

        polygons1 = self.resize_transforms[0].apply_polygons(polygons1)
        polygons2 = self.resize_transforms[1].apply_polygons(polygons2)
        polygons3 = self.resize_transforms[2].apply_polygons(polygons3)
        polygons4 = self.resize_transforms[3].apply_polygons(polygons4)

        # polygons1 in the upper left (no change)
        # polygons2 in the upper right (update x coordinates)
        # polygons3 in the lower left (update y coordinates)
        # polygons4 in the lower right (update both x and y coordinates)
        for p in polygons2:
            p[:, 0] += self.div_x
        for p in polygons3:
            p[:, 1] += self.div_y
        for p in polygons4:
            p[:, 0] += self.div_x
            p[:, 1] += self.div_y

        polygons.extend(polygons1)
        polygons.extend(polygons2)
        polygons.extend(polygons3)
        polygons.extend(polygons4)

        return polygons


class MosaicAugmentation(Augmentation):
    """
    Apply a Mosaic Augmentation to input images.
    """

    def __init__(
        self, output_width: int, output_height: int, div_x: int, div_y: int, interp: int = None
    ):
        """
        Args:
            output_width, output_height (int): shape of output image
            div_x, div_y (int): dividing locations on the x and y axis separating mosaic inputs
            interp (Optional[int]): PIL interp method for resizing.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image1, image2, image3, image4):
        if not len(image1.shape) == len(image2.shape) == len(image3.shape) == len(image4.shape):
            raise ValueError("Image Shapes should be the same.")
        if not image1.dtype == image2.dtype == image3.dtype == image4.dtype:
            raise ValueError("Imge Datatypes should be the same.")

        if len(image1.shape) == 4:
            heights = [image1.shape[1], image2.shape[1], image3.shape[1], image4.shape[1]]
            widths = [image1.shape[2], image2.shape[2], image3.shape[2], image4.shape[2]]
        else:
            heights = [image1.shape[0], image2.shape[0], image3.shape[0], image4.shape[0]]
            widths = [image1.shape[1], image2.shape[1], image3.shape[1], image4.shape[1]]

        return MosaicTransform(
            self.output_width,
            self.output_height,
            self.div_x,
            self.div_y,
            widths,
            heights,
            interp=self.interp,
        )

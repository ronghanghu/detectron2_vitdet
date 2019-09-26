import numpy as np
import cv2
from typing import Optional, Tuple, Iterable

from ..structures import DensePoseDataRelative, DensePoseOutput, DensePoseResult
from .base import MatrixVisualizer, PointsVisualizer, Boxes, Image


class DensePoseResultsVisualizer(object):
    def __init__(
        self,
        data_extractor,
        segm_extractor,
        inplace=True,
        cmap=cv2.COLORMAP_PARULA,
        alpha=0.7,
        val_scale=1.0,
    ):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=val_scale, alpha=alpha
        )
        self.data_extractor = data_extractor
        self.segm_extractor = segm_extractor

    def visualize(
            self,
            image_bgr: Image,
            densepose_result: Optional[DensePoseResult]) -> Image:
        if densepose_result is None:
            return image_bgr
        for i, result_encoded_w_shape in enumerate(densepose_result.results):
            iuv_arr = DensePoseResult.decode_png_data(*result_encoded_w_shape)
            bbox_xywh = densepose_result.boxes_xywh[i]
            matrix = self.data_extractor(iuv_arr)
            segm = self.segm_extractor(iuv_arr)
            mask = np.zeros(matrix.shape, dtype=np.uint8)
            mask[segm > 0] = 1
            image_bgr = self.mask_visualizer.visualize(image_bgr, mask, matrix, bbox_xywh)
        return image_bgr


def _extract_i_from_iuvarr(iuv_arr):
    return iuv_arr[0, :, :]


def _extract_u_from_iuvarr(iuv_arr):
    return iuv_arr[1, :, :]


def _extract_v_from_iuvarr(iuv_arr):
    return iuv_arr[2, :, :]


class DensePoseResultsFineSegmentationVisualizer(DensePoseResultsVisualizer):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7):
        super(DensePoseResultsFineSegmentationVisualizer, self).__init__(
            _extract_i_from_iuvarr,
            _extract_i_from_iuvarr,
            inplace,
            cmap,
            alpha,
            val_scale=255.0 / DensePoseDataRelative.N_PART_LABELS,
        )


class DensePoseResultsUVisualizer(DensePoseResultsVisualizer):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7):
        super(DensePoseResultsUVisualizer, self).__init__(
            _extract_u_from_iuvarr, _extract_i_from_iuvarr, inplace, cmap, alpha, val_scale=1.0
        )


class DensePoseResultsVVisualizer(DensePoseResultsVisualizer):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7):
        super(DensePoseResultsVVisualizer, self).__init__(
            _extract_v_from_iuvarr, _extract_i_from_iuvarr, inplace, cmap, alpha, val_scale=1.0
        )


class DensePoseOutputsFineSegmentationVisualizer(object):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace,
            cmap=cmap,
            val_scale=255.0 / DensePoseDataRelative.N_PART_LABELS,
            alpha=alpha,
        )

    def visualize(
            self,
            image_bgr: Image,
            dp_output_with_bboxes: Optional[Tuple[DensePoseOutput, Boxes]]) \
            -> Image:
        if dp_output_with_bboxes is None:
            return image_bgr
        densepose_output, bboxes_xywh = dp_output_with_bboxes
        S = densepose_output.S
        I = densepose_output.I  # noqa
        U = densepose_output.U
        V = densepose_output.V
        N = S.size(0)
        assert N == I.size(0), (
            "densepose outputs S {} and I {}"
            " should have equal first dim size".format(S.size(), I.size())
        )
        assert N == U.size(0), (
            "densepose outputs S {} and U {}"
            " should have equal first dim size".format(S.size(), U.size())
        )
        assert N == V.size(0), (
            "densepose outputs S {} and V {}"
            " should have equal first dim size".format(S.size(), V.size())
        )
        assert N == len(bboxes_xywh), (
            "number of bounding boxes {}"
            " should be equal to first dim size of outputs {}".format(len(bboxes_xywh), N)
        )
        for n in range(N):
            Sn = S[n].argmax(dim=0)
            In = I[n].argmax(dim=0) * (Sn > 0).long()
            matrix = In.cpu().numpy().astype(np.uint8)
            mask = np.zeros(matrix.shape, dtype=np.uint8)
            mask[matrix > 0] = 1
            bbox_xywh = bboxes_xywh[n]
            image_bgr = self.mask_visualizer.visualize(image_bgr, mask, matrix, bbox_xywh)
        return image_bgr


class DensePoseOutputsUVisualizer(object):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=1.0, alpha=alpha
        )

    def visualize(
            self,
            image_bgr: Image,
            dp_output_with_bboxes: Optional[Tuple[DensePoseOutput, Boxes]]) \
            -> Image:
        if dp_output_with_bboxes is None:
            return image_bgr
        densepose_output, bboxes_xywh = dp_output_with_bboxes
        assert isinstance(
            densepose_output, DensePoseOutput
        ), "DensePoseOutput expected, {} encountered".format(type(densepose_output))
        S = densepose_output.S
        I = densepose_output.I  # noqa
        U = densepose_output.U
        V = densepose_output.V
        N = S.size(0)
        assert N == I.size(0), (
            "densepose outputs S {} and I {}"
            " should have equal first dim size".format(S.size(), I.size())
        )
        assert N == U.size(0), (
            "densepose outputs S {} and U {}"
            " should have equal first dim size".format(S.size(), U.size())
        )
        assert N == V.size(0), (
            "densepose outputs S {} and V {}"
            " should have equal first dim size".format(S.size(), V.size())
        )
        assert N == len(bboxes_xywh), (
            "number of bounding boxes {}"
            " should be equal to first dim size of outputs {}".format(len(bboxes_xywh), N)
        )
        for n in range(N):
            Sn = S[n].argmax(dim=0)
            In = I[n].argmax(dim=0) * (Sn > 0).long()
            segmentation = In.cpu().numpy().astype(np.uint8)
            mask = np.zeros(segmentation.shape, dtype=np.uint8)
            mask[segmentation > 0] = 1
            Un = U[n].cpu().numpy().astype(np.float32)
            Uvis = np.zeros(segmentation.shape, dtype=np.float32)
            for partId in range(Un.shape[0]):
                Uvis[segmentation == partId] = Un[partId][segmentation == partId].clip(0, 1) * 255
                bbox_xywh = bboxes_xywh[n]
            image_bgr = self.mask_visualizer.visualize(image_bgr, mask, Uvis, bbox_xywh)
        return image_bgr


class DensePoseOutputsVVisualizer(object):
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=1.0, alpha=alpha
        )

    def visualize(
            self,
            image_bgr: Image,
            dp_output_with_bboxes: Optional[Tuple[DensePoseOutput, Boxes]]) \
            -> Image:
        if dp_output_with_bboxes is None:
            return image_bgr
        densepose_output, bboxes_xywh = dp_output_with_bboxes
        assert isinstance(
            densepose_output, DensePoseOutput
        ), "DensePoseOutput expected, {} encountered".format(type(densepose_output))
        S = densepose_output.S
        I = densepose_output.I  # noqa
        U = densepose_output.U
        V = densepose_output.V
        N = S.size(0)
        assert N == I.size(0), (
            "densepose outputs S {} and I {}"
            " should have equal first dim size".format(S.size(), I.size())
        )
        assert N == U.size(0), (
            "densepose outputs S {} and U {}"
            " should have equal first dim size".format(S.size(), U.size())
        )
        assert N == V.size(0), (
            "densepose outputs S {} and V {}"
            " should have equal first dim size".format(S.size(), V.size())
        )
        assert N == len(bboxes_xywh), (
            "number of bounding boxes {}"
            " should be equal to first dim size of outputs {}".format(len(bboxes_xywh), N)
        )
        for n in range(N):
            Sn = S[n].argmax(dim=0)
            In = I[n].argmax(dim=0) * (Sn > 0).long()
            segmentation = In.cpu().numpy().astype(np.uint8)
            mask = np.zeros(segmentation.shape, dtype=np.uint8)
            mask[segmentation > 0] = 1
            Vn = V[n].cpu().numpy().astype(np.float32)
            Vvis = np.zeros(segmentation.shape, dtype=np.float32)
            for partId in range(Vn.size(0)):
                Vvis[segmentation == partId] = Vn[partId][segmentation == partId].clip(0, 1) * 255
            bbox_xywh = bboxes_xywh[n]
            image_bgr = self.mask_visualizer.visualize(image_bgr, mask, Vvis, bbox_xywh)
        return image_bgr


class DensePoseDataCoarseSegmentationVisualizer(object):
    """
    Visualizer for ground truth segmentation
    """

    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace,
            cmap=cmap,
            val_scale=255.0 / DensePoseDataRelative.N_BODY_PARTS,
            alpha=alpha,
        )

    def visualize(
            self,
            image_bgr : Image,
            bbox_densepose_datas:
            Optional[Tuple[Iterable[Boxes], Iterable[DensePoseDataRelative]]]) \
            -> Image:
        if bbox_densepose_datas is None:
            return image_bgr
        for bbox_xywh, densepose_data in zip(*bbox_densepose_datas):
            matrix = densepose_data.segm.numpy()
            mask = np.zeros(matrix.shape, dtype=np.uint8)
            mask[matrix > 0] = 1
            image_bgr = self.mask_visualizer.visualize(
                image_bgr, mask, matrix, bbox_xywh.numpy())
        return image_bgr


class DensePoseDataPointsVisualizer(object):
    def __init__(self, densepose_data_to_value_fn=None, cmap=cv2.COLORMAP_PARULA):
        self.points_visualizer = PointsVisualizer()
        self.densepose_data_to_value_fn = densepose_data_to_value_fn
        self.cmap = cmap

    def visualize(
            self,
            image_bgr: Image,
            bbox_densepose_datas:
            Optional[Tuple[Iterable[Boxes], Iterable[DensePoseDataRelative]]]) \
            -> Image:
        if bbox_densepose_datas is None:
            return image_bgr
        for bbox_xywh, densepose_data in zip(*bbox_densepose_datas):
            x0, y0, w, h = bbox_xywh.numpy()
            x = densepose_data.x.numpy() * w / 255.0 + x0
            y = densepose_data.y.numpy() * h / 255.0 + y0
            pts_xy = zip(x, y)
            if self.densepose_data_to_value_fn is None:
                image_bgr = self.points_visualizer.visualize(image_bgr, pts_xy)
            else:
                v = self.densepose_data_to_value_fn(densepose_data)
                img_colors_bgr = cv2.applyColorMap(v, self.cmap)
                colors_bgr = [
                    [int(v) for v in img_color_bgr.ravel()]
                    for img_color_bgr in img_colors_bgr
                ]
                image_bgr = self.points_visualizer.visualize(
                    image_bgr, pts_xy, colors_bgr)
        return image_bgr


def _densepose_data_u_for_cmap(densepose_data):
    u = np.clip(densepose_data.u.numpy(), 0, 1) * 255.0
    return u.astype(np.uint8)


def _densepose_data_v_for_cmap(densepose_data):
    v = np.clip(densepose_data.v.numpy(), 0, 1) * 255.0
    return v.astype(np.uint8)


def _densepose_data_i_for_cmap(densepose_data):
    i = np.clip(
        densepose_data.i.numpy(),
        0.0, DensePoseDataRelative.N_PART_LABELS) * \
        255.0 / DensePoseDataRelative.N_PART_LABELS
    return i.astype(np.uint8)


class DensePoseDataPointsUVisualizer(DensePoseDataPointsVisualizer):
    def __init__(self):
        super(DensePoseDataPointsUVisualizer, self).__init__(
            densepose_data_to_value_fn=_densepose_data_u_for_cmap
        )


class DensePoseDataPointsVVisualizer(DensePoseDataPointsVisualizer):
    def __init__(self):
        super(DensePoseDataPointsVVisualizer, self).__init__(
            densepose_data_to_value_fn=_densepose_data_v_for_cmap
        )


class DensePoseDataPointsIVisualizer(DensePoseDataPointsVisualizer):
    def __init__(self):
        super(DensePoseDataPointsIVisualizer, self).__init__(
            densepose_data_to_value_fn=_densepose_data_i_for_cmap
        )

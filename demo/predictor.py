import cv2
import torch

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.visualizer import ColoringMode, Visualizer


class COCODemo(object):
    def __init__(self, cfg, metadata, confidence_threshold=0.7, stuff_area_threshold=4096):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)  # cfg can be modified by model
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.metadata = metadata

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHT)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST]
        )

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.stuff_area_threshold = stuff_area_threshold

    def run_on_image(self, image, dataset="coco_2017_train"):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (Instances): the detected objects.
            vis_output (VisualizedImageOutput): the visualized image output.
        """
        vis_output = None
        predictions = self.compute_predictions(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image)
        visualizer = Visualizer(image, self.metadata)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg, segments_info, area_limit=self.stuff_area_threshold
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg_predictions(
                    predictions=predictions["sem_seg"].to("cpu"),
                    area_limit=self.stuff_area_threshold,
                    coloring_mode=ColoringMode.SEGMENTATION_FOCUSED,
                )
            if "instances" in predictions:
                predictions = self.select_top_predictions(
                    predictions["instances"].to(self.cpu_device)
                )
                vis_output = visualizer.draw_instance_predictions(predictions=predictions.to("cpu"))

        return predictions, vis_output

    def compute_predictions(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (Instances): the detected objects.
        """
        # apply pre-processing to image
        # whether the model expects BGR inputs or RGB
        if self.cfg.INPUT.FORMAT == "RGB":
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.transform_gen.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        # compute predictions
        with torch.no_grad():
            predictions = self.model([inputs])
        # there is only a single image
        predictions = predictions[0]
        return predictions

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Args:
            predictions (Instances): the result of the computation by the model.

        Returns:
            predictions (Instances): the detected objects.
        """
        scores = predictions.scores
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.scores
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

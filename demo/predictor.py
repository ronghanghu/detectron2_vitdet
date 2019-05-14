import torch

from detectron2.data.transforms import ImageTransformers, Normalize, ResizeShortestEdge
from detectron2.detection.checkpoint import DetectionCheckpointer

# import the hard-coded metadata
from detectron2.detection.data.builtin import COCO_CATEGORIES
from detectron2.detection.modeling import build_model
from detectron2.utils.vis import draw_instance_predictions, draw_stuff_predictions


class COCODemo(object):
    def __init__(self, cfg, confidence_threshold=0.7, stuff_area_threshold=4096):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)  # cfg can be modified by model
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.stuff_area_threshold = stuff_area_threshold

        self.stuff_names = [
            k["name"].replace("-other", "").replace("-merged", "")
            for k in COCO_CATEGORIES
            if k["isthing"] == 0
        ]
        self.things_names = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1] + [
            "__background"
        ]

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg
        transforms = ImageTransformers(
            [
                ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST]),
                Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ]
        )
        return transforms

    def run_on_opencv_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (Instances): the detected objects.
            viz (np.ndarray): the visualization
        """
        predictions = self.compute_predictions(image)
        result = image.copy()
        color_idx = 0
        if "sem_seg" in predictions:
            result, color_idx = draw_stuff_predictions(
                result,
                predictions["sem_seg"].to(self.cpu_device),
                color_idx,
                self.stuff_names,
                self.stuff_area_threshold,
            )
        if "detector" in predictions:
            top_predictions = self.select_top_predictions(
                predictions["detector"].to(self.cpu_device)
            )
            result, color_idx = draw_instance_predictions(
                result, top_predictions, color_idx, self.things_names
            )
        return top_predictions, result

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
        image = self.transforms.transform_image(original_image)
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

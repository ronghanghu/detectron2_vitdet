import cv2
import pycocotools.mask as mask_util
import torch

from detectron2.data.transforms import ImageTransformers, Normalize, ResizeShortestEdge
from detectron2.detection.checkpoint import DetectionCheckpointer
from detectron2.detection.coco_evaluation import postprocess
from detectron2.detection.modeling import build_detection_model
from detectron2.structures.image_list import ImageList


class COCODemo(object):
    # COCO categories for pretty print
    CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(self, cfg, confidence_threshold=0.7):
        self.cfg = cfg.clone()
        self.model = build_detection_model(self.cfg)  # cfg can be modified by model
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        self.mask_threshold = 0.5
        self.padding = 1

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

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
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        result = self.overlay_boxes(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)

        return top_predictions, result

    def compute_predictions(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (Instances): the detected objects.
        """
        # apply pre-processing to image
        if not self.cfg.INPUT.BGR:  # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        image = self.transforms.transform_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        # convert to an ImageList, padded so that it is divisible by cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = ImageList.from_tensors([image], self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model((image_list, None, None))
        # there is only a single image
        predictions = predictions[0].to(self.cpu_device)

        # reshape prediction (a Boxes) into the original image size
        height, width = original_image.shape[:2]
        predictions = postprocess(predictions, width, height)

        if predictions.has("pasted_mask_rle"):
            predictions.pred_masks = torch.as_tensor(
                [mask_util.decode(k) for k in predictions.pasted_mask_rle]
            )
            predictions.remove("pasted_mask_rle")
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

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Draw the predicted boxes on top of the image

        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            predictions (Instances): the result of the computation by the model.
                It should contain the field "pred_classes", "pred_boxes".

        Returns:
            The image with boxes on it.
        """
        labels = predictions.pred_classes
        boxes = predictions.pred_boxes.tensor

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), tuple(color), 1)

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            predictions (Instances): the result of the computation by the model.
                It should contain the field "pred_masks" and "pred_classes".

        Returns:
            The image with masks on it.
        """
        masks = predictions.pred_masks.numpy()
        labels = predictions.pred_classes

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[:, :, None]
            outputs = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = outputs[-2:]  # opencv func signature has changed between versions
            image = cv2.drawContours(image, contours, -1, color, 3)

        return image

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            predictions (Instances): the result of the computation by the model.
                It should contain the field "scores" and "pred_classes".

        Returns:
            The image with texts on it.
        """
        scores = predictions.scores.tolist()
        labels = predictions.pred_classes.tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.pred_boxes.tensor

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image

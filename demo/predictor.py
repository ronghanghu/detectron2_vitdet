import os
import cv2
import torch

import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColoringMode, Visualizer


class COCODemo(object):
    def __init__(self, cfg, stuff_area_threshold=4096):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)  # cfg can be modified by model
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHT)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST]
        )

        self.cpu_device = torch.device("cpu")
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"]
        self.stuff_area_threshold = stuff_area_threshold

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.compute_predictions(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = torch.tensor(image).flip([2])
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
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(
                    predictions=instances, coloring_mode=ColoringMode.IMAGE_FOCUSED
                )

        return predictions, vis_output

    def run_on_video(self, input_path, output_path=None):
        """
        Visualizes predictions on frames of the input video. Creates and saves the output
        video to the output file path, if provided. Otherwise, displays the visualized
        output video in a window.

        Args:
            input_path (str): file path for input video file.
            output_path (str, optional): file path for visualized output video file.
        """
        assert os.path.isfile(input_path)
        video = cv2.VideoCapture(input_path)

        # frequency of frame capture
        seconds = 0.03
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        frame_grab_frequency = int(round(seconds * frames_per_second))

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if output_path:
            video_file = cv2.VideoWriter(
                filename=output_path,
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )

        video_visualizer = VideoVisualizer(self.metadata)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame_number = int(round(video.get(cv2.CAP_PROP_POS_FRAMES)))
                if frame_number % frame_grab_frequency == 0:
                    predictions = self.compute_predictions(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.tensor(frame)

                    vis_frame = None
                    if "instances" in predictions:
                        predictions = predictions["instances"].to(self.cpu_device)
                        vis_frame = video_visualizer.draw_instance_predictions(
                            frame=frame, predictions=predictions.to("cpu")
                        )

                    # Converts Matplotlib RGB format to OpenCV BGR format before visualizing
                    # output in window.
                    vis_frame = vis_frame.get_image()[:, :, [2, 1, 0]]
                    vis_frame = cv2.resize(vis_frame, (width, height))
                    if output_path:
                        video_file.write(vis_frame)
                    else:
                        cv2.imshow("COCO detections", vis_frame)
                        if cv2.waitKey(1) == 27:
                            break  # esc to quit
            else:
                break

        video.release()
        if output_path:
            video_file.release()
        cv2.destroyAllWindows()

    @torch.no_grad()
    def compute_predictions(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (Instances): the detected objects.
        """
        # apply pre-processing to image
        # whether the model expects BGR inputs or RGB
        if self.input_format == "RGB":
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.transform_gen.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
        return predictions

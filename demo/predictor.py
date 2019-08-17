import torch

from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class VisualizationDemo(object):
    def __init__(self, cfg, stuff_area_threshold=4096, instance_mode=ColorMode.IMAGE):
        self.predictor = DefaultPredictor(cfg)
        self.metadata = self.predictor.metadata
        self.cpu_device = torch.device("cpu")
        self.stuff_area_threshold = stuff_area_threshold
        self.instance_mode = instance_mode

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
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device),
                segments_info,
                area_limit=self.stuff_area_threshold,
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg_predictions(
                    predictions=predictions["sem_seg"].argmax(dim=0).to(self.cpu_device),
                    area_limit=self.stuff_area_threshold,
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)
        for _, frame in enumerate(self._frame_from_video(video)):
            predictions = self.predictor(frame)
            frame = frame[:, :, ::-1]
            vis_frame = None
            if "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            # TODO semantic / panoptic

            # Converts Matplotlib RGB format to OpenCV BGR format before visualizing
            # output in window.
            vis_frame = vis_frame.get_image()[:, :, ::-1]
            yield vis_frame

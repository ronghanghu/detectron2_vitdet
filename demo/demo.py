from detectron2.utils.env import setup_environment  # noqa F401  isort:skip

import argparse
import glob
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger

from predictor import COCODemo


def main():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(cfg, metadata, confidence_threshold=args.confidence_threshold)

    if args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"

        cam = cv2.VideoCapture(0)
        while True:
            start_time = time.time()
            ret_val, img = cam.read()
            predictions, visualized_output = coco_demo.run_on_image(img)
            logger.info(
                "Time: {:.2f} s; {} instances detected".format(time.time() - start_time),
                len(predictions),
            )
            if visualized_output:
                # Converts Matplotlib RGB format to OpenCV BGR format before visualizing output.
                cv2.imshow("COCO detections", visualized_output.get_image()[:, :, [2, 1, 0]])
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()

    elif args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
        for path in tqdm.tqdm(args.input, disable=not args.output):
            img = cv2.imread(path)
            start_time = time.time()
            predictions, visualized_output = coco_demo.run_on_image(img)
            logger.info(
                "{}: detected {} instances in {:.1f}s".format(
                    path, len(predictions), time.time() - start_time
                )
            )

            if args.output:
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
                if visualized_output:
                    visualized_output.save(out_filename)
            else:
                if visualized_output:
                    # Converts Matplotlib RGB format to OpenCV BGR format before visualizing output.
                    cv2.imshow("COCO detections", visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit

    elif args.video_input:
        if args.output:
            video_output_filename = "visualized-" + os.path.basename(args.video_input)
            video_output = os.path.join(args.output, video_output_filename)
        else:
            video_output = None
        coco_demo.run_on_video(args.video_input, video_output)


if __name__ == "__main__":
    main()

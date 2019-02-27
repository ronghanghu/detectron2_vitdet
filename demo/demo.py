from detectron2.utils.env import setup_environment  # noqa F401  isort:skip

import argparse
import os
import time
import cv2

from detectron2.detection import get_cfg, set_global_cfg

from predictor import COCODemo


def main():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
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

    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg.GLOBAL)

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(cfg, confidence_threshold=args.confidence_threshold)

    if args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"

        cam = cv2.VideoCapture(0)
        while True:
            start_time = time.time()
            ret_val, img = cam.read()
            results, composite = coco_demo.run_on_opencv_image(img)
            print("#instances: ", len(results))
            print("Time: {:.2f} s / img".format(time.time() - start_time))
            cv2.imshow("COCO detections", composite)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    else:
        for path in args.input:
            img = cv2.imread(path)
            start_time = time.time()
            results, composite = coco_demo.run_on_opencv_image(img)
            print("#instances: ", len(results))
            print("Time: {:.2f} s / img".format(time.time() - start_time))

            if args.output:
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
                cv2.imwrite(out_filename, composite)
            else:
                cv2.imshow("COCO detections", composite)
                if cv2.waitKey(0) == 27:
                    break  # esc to quit


if __name__ == "__main__":
    main()

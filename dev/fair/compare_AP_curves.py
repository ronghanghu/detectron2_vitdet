import argparse
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from detectron2.data import MetadataCatalog
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", help="path to the prediction json files")
    parser.add_argument("--names", nargs="+", help="short name for each prediction")
    parser.add_argument("--dataset", default="coco_2017_val", help="dataset name to evalaute")
    parser.add_argument(
        "--iou_type",
        default="bbox",
        help="COCO API eval iou_type",
        choices=["bbox", "segm", "keypoints"],
    )
    parser.add_argument(
        "--area",
        default=["all"],
        help="list of area rng to consider",
        choices=["all", "small", "medium", "large"],
        nargs="+",
    )
    parser.add_argument("--output", default="output", help="output directory")
    args = parser.parse_args()

    if args.names is None:
        args.names = [os.path.basename(x)[:-5] for x in args.predictions]
    if not len(args.predictions):
        raise ValueError("--predictions must contain a list of files")
    if len(args.names) != len(args.predictions):
        raise ValueError("--names must match the length of --predictions")

    logger = setup_logger()

    metadata = MetadataCatalog.get(args.dataset)
    coco_api = COCO(PathManager.get_local_path(metadata.json_file))

    precision_per_prediction = []

    for fname in args.predictions:
        with PathManager.open(fname) as f:
            coco_results = json.load(f)
        coco_eval = _evaluate_predictions_on_coco(coco_api, coco_results, args.iou_type)
        precision = coco_eval.eval["precision"]
        precision_per_prediction.append(precision)

    PathManager.mkdirs(args.output)

    num_iou, num_recall, num_cls, num_area = precision_per_prediction[0].shape[:4]
    recalls = coco_eval.params.recThrs
    ious = coco_eval.params.iouThrs
    area_labels = coco_eval.params.areaRngLbl

    for cls in range(num_cls):
        clsname = metadata.thing_classes[cls]
        logger.info(f"Processing class={cls}:{clsname}")
        for area_label in args.area:
            area_idx = area_labels.index(area_label)

            assert num_iou == 10
            rows, cols = 3, 4  # 3x4=12 > 10
            fig, axes_matrix = plt.subplots(
                nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=(cols * 3, rows * 3.3)
            )
            fig.suptitle(f"Class={cls}:{clsname}; Area={area_label}")
            plt.subplots_adjust(hspace=0.3)
            for iou_idx, iou in enumerate(ious):
                irow, icol = iou_idx // cols, iou_idx % cols
                ax = axes_matrix[irow, icol]

                APs = []
                for precision in precision_per_prediction:
                    # -1: maxdet=100
                    ys = precision[iou_idx, :, cls, area_idx, -1]
                    AP = np.mean(ys) * 100
                    ax.plot(recalls, ys)
                    APs.append(AP)
                title = f"IoU={iou:.2f};  AP per file:\n" + "\n".join(
                    [f"{name}={AP:.1f}" for name, AP in zip(args.names, APs)]
                )

                ax.set_title(title, fontsize=7)
                ax.set_aspect(1.0)
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.set_xticks(np.arange(0.0, 1.2, 0.2))
                ax.set_yticks(np.arange(0.0, 1.2, 0.2))
                if irow == rows - 1:
                    ax.set_xlabel("Recall")
                if icol == 0:
                    ax.set_ylabel("Precision")
            for iou_idx in range(num_iou, rows * cols):
                ax = axes_matrix[iou_idx // cols, iou_idx % cols]
                ax.set_axis_off()

            fig.savefig(f"{args.output}/{clsname}-{area_label}.pdf")

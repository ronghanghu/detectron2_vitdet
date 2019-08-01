import argparse
import os

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.utils.env import setup_environment
from detectron2.utils.visualizer import Visualizer


def setup(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualizes Ground-truth Dataset")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    """
    General utility to visualize ground truth dataset.
    """
    setup_environment()
    args = parse_args()
    cfg = setup(args)
    train_data_loader = build_detection_train_loader(cfg)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    for batch in train_data_loader:
        for per_image in batch:
            # Pytorch tensor images are in (C, H, W) format and is reformatted to match
            # Matplotlib's (H, W, C) format. H, W, and C correspond to the image's height,
            # width and number of color channels.
            img = per_image["image"].permute(1, 2, 0)
            # Convert image from BGR format to RGB format.
            img = img[:, :, [2, 1, 0]]

            visualizer = Visualizer(img, metadata=metadata)
            target_fields = per_image["targets"].get_fields()
            vis = visualizer.overlay_instances(
                labels=target_fields.get("gt_classes", None),
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )
            filepath = os.path.join(dirname, str(per_image["image_id"]) + ".jpg")
            print("Saving to {} ...".format(filepath))
            vis.save_output_file(filepath)

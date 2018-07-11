import argparse

import torch

from torch_detectron.core.inference import inference
from torch_detectron.helpers.config_utils import load_config
from torch_detectron.helpers.config_utils import update_config_with_args
from torch_detectron.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch/configs/faster_rcnn_r50_relaunch.py",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--checkpoint", default="", metavar="FILE", help="path to checkpoint file"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    config = load_config(args.config_file)
    update_config_with_args(config, args.opts)

    save_dir = ""
    local_rank = 0
    setup_logger("torch_detectron", save_dir, local_rank)

    # TODO this doesn't look great
    config.TEST.DATA.DATALOADER.SAMPLER.DISTRIBUTED = False
    data_loader_val = config.TEST.DATA()

    device = config.DEVICE
    model = config.MODEL()
    model.to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint["model"])
        model = model.module

    iou_types = ("bbox",)
    if config.MODEL.USE_MASK:
        iou_types = iou_types + ("segm",)
    inference(
        model, data_loader_val, iou_types=iou_types, box_only=config.MODEL.RPN_ONLY
    )


if __name__ == "__main__":
    main()

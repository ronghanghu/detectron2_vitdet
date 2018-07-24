import argparse
import os

import torch

from torch_detectron.core.inference import inference
from torch_detectron.core.utils import load_state_dict
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
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    args.distributed = (
        int(os.environ["WORLD_SIZE"]) > 1 if "WORLD_SIZE" in os.environ else False
    )

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    config = load_config(args.config_file)
    update_config_with_args(config, args.opts)

    save_dir = ""
    setup_logger("torch_detectron", save_dir, args.local_rank)

    # TODO this doesn't look great
    if args.distributed:
        # TODO distributed sampler doesn't support shuffle=False
        config.TEST.DATA.DATALOADER.SAMPLER.SHUFFLE = True
    config.TEST.DATA.DATALOADER.SAMPLER.DISTRIBUTED = args.distributed
    data_loader_val = config.TEST.DATA()
    print('size', len(data_loader_val), args.distributed)

    device = config.DEVICE
    model = config.MODEL()
    model.to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        # TODO find a better way of serializing the weights
        # that avoids this ugly workaround
        model = torch.nn.DataParallel(model)
        load_state_dict(model, checkpoint["model"])
        model = model.module

    iou_types = ("bbox",)
    if config.MODEL.USE_MASK:
        iou_types = iou_types + ("segm",)
    inference(
        model, data_loader_val, iou_types=iou_types, box_only=config.MODEL.RPN_ONLY
    )


if __name__ == "__main__":
    main()

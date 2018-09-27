# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from torch_detectron.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from torch_detectron.config import cfg
from torch_detectron.config.data import make_data_loader
from torch_detectron.engine.inference import inference
from torch_detectron.engine.logger import setup_logger
from torch_detectron.modeling.model_builder import build_detection_model
from torch_detectron.modeling.model_serialization import load_model_file


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
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

    distributed = (
        int(os.environ["WORLD_SIZE"]) > 1 if "WORLD_SIZE" in os.environ else False
    )

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("torch_detectron", save_dir, args.local_rank)
    logger.info(cfg)

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    load_model_file(cfg, model, args.checkpoint)

    data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    inference(
        model,
        data_loader_val,
        iou_types=iou_types,
        box_only=cfg.MODEL.RPN_ONLY,
        device=cfg.MODEL.DEVICE,
    )


if __name__ == "__main__":
    main()

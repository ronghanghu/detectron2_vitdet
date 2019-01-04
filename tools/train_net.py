"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import sys

import torch

from maskrcnn_benchmark.detection import DetectionCheckpointer
from maskrcnn_benchmark.detection import build_detection_model
from maskrcnn_benchmark.detection import coco_evaluation
from maskrcnn_benchmark.detection import get_cfg
from maskrcnn_benchmark.detection import make_detection_data_loader
from maskrcnn_benchmark.detection import make_lr_scheduler
from maskrcnn_benchmark.detection import make_optimizer
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.engine.launch import launch
from maskrcnn_benchmark.utils.collect_env import collect_env_info
import maskrcnn_benchmark.utils.comm as comm
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def test(cfg, model):
    distributed = comm.get_world_size() > 1
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    if cfg.OUTPUT_DIR:
        dataset_names = cfg.DATASETS.TEST
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_detection_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
        coco_evaluation(
            model,
            data_loader_val,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            output_folder=output_folder,
        )
        comm.synchronize()


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger(output_dir, distributed_rank=comm.get_rank())
    logger.info("Using {} GPUs".format(args.num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    return cfg


def main(args):
    cfg = setup(args)
    model = build_detection_model(cfg)
    output_dir = cfg.OUTPUT_DIR

    if args.eval_only:
        checkpointer = DetectionCheckpointer(cfg, model, save_dir=output_dir)
        checkpointer.load(cfg.MODEL.WEIGHT)
        test(cfg, model)
        return

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    distributed = comm.get_world_size() > 1
    if distributed:
        local_rank = comm.get_rank() % args.num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    extra_checkpoint_data = {"iteration": 0}

    save_to_disk = comm.get_rank() == 0
    checkpointer = DetectionCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk)
    extra_checkpoint_data.update(checkpointer.load(cfg.MODEL.WEIGHT))

    data_loader = make_detection_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=extra_checkpoint_data["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        checkpoint_period,
        extra_checkpoint_data,
    )

    test(cfg, model.module if distributed else model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:12456")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    launch(main, args.num_gpus, dist_url=args.dist_url, args=(args,))

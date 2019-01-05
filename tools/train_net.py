"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import logging
import datetime
import time
import argparse
import os

import torch

from maskrcnn_benchmark.detection import DetectionCheckpointer
from maskrcnn_benchmark.detection import build_detection_model
from maskrcnn_benchmark.detection import coco_evaluation
from maskrcnn_benchmark.detection import get_cfg
from maskrcnn_benchmark.detection import make_detection_data_loader
from maskrcnn_benchmark.detection import make_lr_scheduler
from maskrcnn_benchmark.detection import make_optimizer
from maskrcnn_benchmark.engine.launch import launch
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import reduce_dict
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
import maskrcnn_benchmark.utils.comm as comm


class PeriodicCheckpointer(object):
    def __init__(self, checkpointer, period, max_iter):
        self.checkpointer = checkpointer
        self.period = period
        self.max_iter = max_iter

    def step(self, iteration, **kwargs):
        """
        Args:
            kwargs: extra data to save, same as in :meth:`Checkpointer.save`
        """
        if iteration % self.period == 0:
            self.checkpointer.save("model_{:07d}".format(iteration), iteration=iteration)
        if iteration == self.max_iter:
            self.checkpointer.save("model_final", iteration=iteration)


def do_test(cfg, model):
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


def do_train(
    model, data_loader, optimizer, scheduler, periodic_checkpointer, start_iter=0
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    model.train()
    start_training_time = time.time()
    iter_end = time.time()
    for iteration, data in enumerate(data_loader, start_iter):
        data_time = time.time() - iter_end
        iteration = iteration + 1

        scheduler.step()

        loss_dict = model(data)

        losses = sum(loss for loss in loss_dict.values())
        if torch.isnan(losses).any():
            raise FloatingPointError("Losses become NaN at iteration={}!".format(iteration))

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - iter_end
        iter_end = time.time()
        # Consider the first several iteration as warmup and don't time it.
        if iteration <= 3:
            start_training_time = time.time()
        else:
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # TODO Move logging logic to a centralized system (https://github.com/fairinternal/detectron2/issues/9)
        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                # NOTE this format is parsed by grep
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        periodic_checkpointer.step(iteration)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


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
        do_test(cfg, model)
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

    checkpointer = DetectionCheckpointer(cfg, model, optimizer, scheduler, output_dir)
    start_iter = checkpointer.load(cfg.MODEL.WEIGHT).get("iteration", 0)

    data_loader = make_detection_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=start_iter,
    )

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, cfg.SOLVER.MAX_ITER),
        start_iter=start_iter,
    )

    do_test(cfg, model.module if distributed else model)


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

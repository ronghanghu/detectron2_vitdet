"""
Detection Training Script.
TODO rename the file.
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import datetime
import logging
import os
import time

import torch
from torch.nn.parallel import DistributedDataParallel

import maskrcnn_benchmark.utils.comm as comm
from maskrcnn_benchmark.detection import (
    DetectionCheckpointer,
    build_detection_model,
    coco_evaluation,
    print_copypaste_format,
    get_cfg,
    make_detection_data_loader,
    make_lr_scheduler,
    make_optimizer,
    verify_results
)
from maskrcnn_benchmark.engine.launch import launch
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import reduce_dict
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


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


def do_test(cfg, model, is_final=True):
    """
    Args:
        is_final (bool): if True, will print results in a copy-paste friendly
            format and will run verification.

    Returns:
        list[result]: only on the main process, result for each DATASETS.TEST. Each result is a dict of
            dict. result[task][metric] is a float.
    """
    if isinstance(model, DistributedDataParallel):
        model = model.module
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
    data_loaders_val = make_detection_data_loader(
        cfg, is_train=False, is_distributed=comm.get_world_size() > 1
    )

    results = []
    for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
        results_per_dataset = coco_evaluation(
            model,
            data_loader_val,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            output_folder=output_folder,
        )
        if comm.is_main_process():
            results.append(results_per_dataset[0])
            if is_final:
                print_copypaste_format(results_per_dataset[0])
        comm.synchronize()
    model.train()  # model is set to eval mode by `coco_evaluation`

    if is_final and cfg.TEST.EXPECTED_RESULTS and comm.is_main_process():
        assert len(results) == 1, "Results verification only supports one dataset!"
        verify_results(cfg, results[0])
    return results


def do_train(cfg, model):
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(cfg, model, optimizer, scheduler, cfg.OUTPUT_DIR)
    start_iter = checkpointer.load(cfg.MODEL.WEIGHT).get("iteration", 0)
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, cfg.SOLVER.MAX_ITER
    )

    data_loader = make_detection_data_loader(
        cfg, is_train=True, is_distributed=comm.get_world_size() > 1, start_iter=start_iter
    )
    assert len(data_loader) == cfg.SOLVER.MAX_ITER

    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = cfg.SOLVER.MAX_ITER
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

        if cfg.TEST.EVAL_PERIOD > 0 and iteration % cfg.TEST.EVAL_PERIOD == 0:
            do_test(cfg, model, is_final=False)

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

    logger.info(
        "Loaded config file {}:\n{}".format(args.config_file, open(args.config_file, "r").read())
    )
    logger.info("Running with full config:\n{}".format(cfg))
    if comm.get_rank() == 0 and output_dir:
        path = os.path.join(output_dir, "config.yaml")
        with open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))
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

    distributed = comm.get_world_size() > 1
    if distributed:
        local_rank = comm.get_rank() % args.num_gpus
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    do_train(cfg, model)
    do_test(cfg, model)


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

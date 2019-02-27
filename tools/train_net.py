"""
Detection Training Script.
TODO rename the file.
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from detectron2.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import datetime
import logging
import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.detection import (
    DetectionCheckpointer,
    build_detection_model,
    build_detection_train_loader,
    build_lr_scheduler,
    build_optimizer,
    coco_evaluation,
    get_cfg,
    print_copypaste_format,
    set_global_cfg,
    verify_results,
)
from detectron2.engine.launch import launch
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.comm import reduce_dict
from detectron2.utils.events import EventStorage, JSONWriter, get_event_storage
from detectron2.utils.logger import setup_logger
from detectron2.utils.misc import mkdir


class PeriodicCheckpointer(object):
    def __init__(self, checkpointer, cfg, period, max_iter):
        self.checkpointer = checkpointer
        self.period = period
        self.max_iter = max_iter
        self.cfg = cfg.clone()

    def step(self, iteration, **kwargs):
        """
        Args:
            kwargs: extra data to save, same as in :meth:`Checkpointer.save`
        """
        additional_state = {"iteration": iteration, "cfg": self.cfg.dump()}
        if iteration % self.period == 0:
            self.checkpointer.save("model_{:07d}".format(iteration), **additional_state)
        if iteration == self.max_iter:
            self.checkpointer.save("model_final", **additional_state)


class MetricPrinter:
    def __init__(self, max_iter):
        self.logger = logging.getLogger("detectron2.trainer")
        self._max_iter = max_iter

    def write(self):
        storage = get_event_storage()
        iteration = storage.iteration
        eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # NOTE: max mem is parsed by grep
        self.logger.info(
            """\
eta: {eta}  iter: {iter}  {losses}  \
time: {time:.4f} data_time: {data_time:.4f}  \
lr: {lr:.6f}  max mem: {memory:.0f}M \
""".format(
                eta=eta_string,
                iter=iteration,
                losses="  ".join(
                    [
                        "{}: {:.3f}".format(k, v.median(20))
                        for k, v in storage.histories().items()
                        if "loss" in k
                    ]
                ),
                time=storage.history("time").global_avg(),
                data_time=storage.history("data_time").median(20),
                lr=storage.history("lr").latest(),
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
            )
        )


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

    results = []
    for dataset_name in cfg.DATASETS.TEST:
        if cfg.OUTPUT_DIR:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
        else:
            output_folder = None

        results_per_dataset = coco_evaluation(cfg, model, dataset_name, output_folder=output_folder)
        if comm.is_main_process():
            results.append(results_per_dataset[0])
            if is_final:
                print_copypaste_format(results_per_dataset[0])
        comm.synchronize()

    if is_final and cfg.TEST.EXPECTED_RESULTS and comm.is_main_process():
        assert len(results) == 1, "Results verification only supports one dataset!"
        verify_results(cfg, results[0])
    return results


def do_train(cfg, model):
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(model, optimizer, scheduler, cfg.OUTPUT_DIR)
    start_iter = checkpointer.load(cfg.MODEL.WEIGHT).get("iteration", 0)
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg, cfg.SOLVER.CHECKPOINT_PERIOD, cfg.SOLVER.MAX_ITER
    )

    data_loader = build_detection_train_loader(cfg, start_iter=start_iter)

    logger = logging.getLogger("detectron2.trainer")
    logger.info("Start training")
    max_iter = cfg.SOLVER.MAX_ITER
    model.train()
    start_training_time = time.time()
    iter_end = time.time()

    writers = (
        [MetricPrinter(max_iter), JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json"))]
        if comm.is_main_process()
        else []
    )
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            data_time = time.time() - iter_end
            iteration = iteration + 1
            storage.step()

            scheduler.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            loss_dict = model(data)

            losses = sum(loss for loss in loss_dict.values())
            if torch.isnan(losses).any():
                raise FloatingPointError("Losses become NaN at iteration={}!".format(iteration))

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            storage.put_scalars(loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_time = time.time() - iter_end
            iter_end = time.time()
            # Consider the first several iteration as warmup and don't time it.
            if iteration <= 3:
                start_training_time = time.time()
            else:
                storage.put_scalars(time=batch_time, data_time=data_time)

            if cfg.TEST.EVAL_PERIOD > 0 and iteration % cfg.TEST.EVAL_PERIOD == 0:
                do_test(cfg, model, is_final=False)

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
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
    set_global_cfg(cfg.GLOBAL)

    colorful_logging = not args.no_color
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger(output_dir, color=colorful_logging, distributed_rank=comm.get_rank())
    logger.info(
        "Using {} GPUs per machine. Rank of current process: {}".format(
            args.num_gpus, comm.get_rank()
        )
    )
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
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    output_dir = cfg.OUTPUT_DIR

    if args.eval_only:
        checkpointer = DetectionCheckpointer(model, save_dir=output_dir)
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
    parser.add_argument("--no-color", action="store_true", help="disable colorful logging")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus per machine")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

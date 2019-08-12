"""
DensePose Training Script.

This script is similar to the training script in detectron2/tools.

It is an example of how a user might use detectron2 for a new project.
"""

import logging
import os
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import SimpleTrainer, default_argument_parser, hooks, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    inference_context,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from densepose import DatasetMapper, DensePoseCOCOEvaluator, add_densepose_config


def get_evaluator(cfg, dataset_name, output_folder):
    evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
    if cfg.MODEL.DENSEPOSE_ON:
        evaluators.append(DensePoseCOCOEvaluator(dataset_name, True, output_folder))
    return DatasetEvaluators(evaluators)


def do_test(cfg, model, is_final=True):
    """
    Args:
        is_final (bool): if True, will print results in a copy-paste friendly
            format and will run verification.

    Returns:
        list[result]: only on the main process, result for each DATASETS.TEST.
            Each result is a dict of dict. result[task][metric] is a float.
    """
    assert len(cfg.DATASETS.TEST)
    if isinstance(model, DistributedDataParallel):
        model = model.module

    assert len(cfg.DATASETS.TEST) == 1, cfg.DATASETS.TEST
    with inference_context(model):
        dataset_name = cfg.DATASETS.TEST[0]
        if cfg.OUTPUT_DIR:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            if comm.is_main_process():
                PathManager.mkdirs(output_folder)
            comm.synchronize()
        else:
            output_folder = None

        data_loader = build_detection_test_loader(
            cfg, dataset_name, mapper=DatasetMapper(cfg, False)
        )
        evaluator = get_evaluator(cfg, dataset_name, output_folder)
        results = inference_on_dataset(model, data_loader, evaluator)
        if comm.is_main_process() and is_final:
            print_csv_format(results)

    if is_final and cfg.TEST.EXPECTED_RESULTS and comm.is_main_process():
        verify_results(cfg, results)
    return results


def create_after_step_hook(cfg, model, optimizer, scheduler, periodic_checkpointer):
    """
    Create a hook that performs some pre-defined tasks used in this script
    (evaluation, LR scheduling, checkpointing).
    """

    def after_step_callback(trainer):
        if (
            cfg.TEST.EVAL_PERIOD > 0
            and (trainer.iter + 1) % cfg.TEST.EVAL_PERIOD == 0
            and trainer.iter != trainer.max_iter - 1
        ):
            results = do_test(cfg, model, is_final=False)

            for dataset_name, results_per_dataset in results.items():
                for task, metrics_per_task in results_per_dataset.items():
                    for metric, value in metrics_per_task.items():
                        key = "{}/{}/{}".format(dataset_name, task, metric)
                        trainer.storage.put_scalar(key, value, smoothing_hint=False)
            # Evaluation may take different time among workers.
            # A barrier make them start the next iteration together.
            comm.synchronize()

        trainer.storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
        scheduler.step()
        periodic_checkpointer.step(trainer.iter)

    return hooks.CallbackHook(after_step=after_step_callback)


def do_train(cfg, model, resume=True):
    """
    Args:
        resume (bool): whether to attemp to resume from checkpoint directory.
            Defaults to True to maintain backward compatibility.
    """
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(model, optimizer, scheduler, cfg.OUTPUT_DIR)
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHT, resume=resume).get("iteration", -1)
    # The checkpoint stores the training iteration that just finished, thus we start
    # at the next iteration (or iter zero if there's no checkpoint).
    start_iter += 1
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    data_loader = build_detection_train_loader(
        cfg, mapper=DatasetMapper(cfg, True), start_iter=start_iter
    )

    trainer = SimpleTrainer(model, data_loader, optimizer)
    trainer_hooks = [
        hooks.IterationTimer(),
        create_after_step_hook(cfg, model, optimizer, scheduler, periodic_checkpointer),
    ]
    if comm.is_main_process():
        writers = [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        trainer_hooks.append(hooks.PeriodicWriter(writers))
    trainer.register_hooks(trainer_hooks)
    trainer.train(start_iter, max_iter)


def setup(args):
    """
    Create configs and setup logger from arguments and the given config file.
    """
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)
    comm.synchronize()

    logger = setup_logger(output_dir, distributed_rank=comm.get_rank())
    logger.info(
        "Using {} GPUs per machine. Rank of current process: {}".format(
            args.num_gpus, comm.get_rank()
        )
    )
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    with PathManager.open(args.config_file, "r") as f:
        logger.info("Loaded config file {}:\n{}".format(args.config_file, f.read()))

    if comm.is_main_process() and output_dir:
        # Other scripts may expect the name config.yaml and depend on this.
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))

    if args.eval_only:
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(cfg.MODEL.WEIGHT, resume=args.resume)
        return do_test(cfg, model)

    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

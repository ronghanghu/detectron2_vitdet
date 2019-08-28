"""
DensePose Training Script.

This script is similar to the training script in detectron2/tools.

It is an example of how a user might use detectron2 for a new project.
"""

import logging
import os
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import SimpleTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    flatten_results_dict,
    inference_context,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    JSONWriter,
    TensorboardXWriter,
    get_event_storage,
)

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
    assert len(cfg.DATASETS.TEST) == 1, cfg.DATASETS.TEST
    with inference_context(model):
        dataset_name = cfg.DATASETS.TEST[0]
        if cfg.OUTPUT_DIR:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        else:
            output_folder = None

        data_loader = build_detection_test_loader(
            cfg, dataset_name, mapper=DatasetMapper(cfg, False)
        )
        evaluator = get_evaluator(cfg, dataset_name, output_folder)
        results = inference_on_dataset(model, data_loader, evaluator)
        if comm.is_main_process():
            if is_final:
                print_csv_format(results)
                verify_results(cfg, results)

            try:
                storage = get_event_storage()
            except Exception:
                pass
            else:
                storage.put_scalars(**flatten_results_dict(results), smoothing_hint=False)
    return results


def create_eval_hook(cfg, model):
    def after_step_callback(trainer):
        if cfg.TEST.EVAL_PERIOD > 0 and (trainer.iter + 1) % cfg.TEST.EVAL_PERIOD == 0:
            do_test(cfg, model, is_final=trainer.iter + 1 == trainer.max_iter)
            # Evaluation may take different time among workers.
            # A barrier make them start the next iteration together.
            comm.synchronize()

    return hooks.CallbackHook(after_step=after_step_callback)


def do_train(cfg, model, resume=True):
    """
    Args:
        resume (bool): whether to attemp to resume from checkpoint directory.
            Defaults to True to maintain backward compatibility.
    """
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    # The checkpoint stores the training iteration that just finished, thus we start
    # at the next iteration (or iter zero if there's no checkpoint).
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHT, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    data_loader = build_detection_train_loader(
        cfg, mapper=DatasetMapper(cfg, True), start_iter=start_iter
    )

    trainer = SimpleTrainer(model, data_loader, optimizer)
    trainer_hooks = [
        hooks.IterationTimer(),
        hooks.LRScheduler(optimizer, scheduler),
        create_eval_hook(cfg, model),
    ]
    if comm.is_main_process():
        writers = [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        trainer_hooks.append(hooks.PeriodicWriter(writers))
        trainer_hooks.append(hooks.PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
    trainer.register_hooks(trainer_hooks)
    trainer.train(start_iter, max_iter)


def setup(args):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHT, resume=args.resume
        )
        return do_test(cfg, model)

    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    return do_train(cfg, model, args.resume)


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

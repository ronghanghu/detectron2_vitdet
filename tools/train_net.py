"""
Detection Training Script.

This scripts reads a given config file and runs the training.
It is an entry point that is made to train all standard models in detectron2.

In order to let one script support training of all the models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a fixed "evaluator",
and doesn't need results verification.

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
from borc.nn.precise_bn import get_bn_modules, update_bn_stats
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, set_global_cfg  # noqa
from detectron2.data import (
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import SimpleTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    flatten_results_dict,
    inference_context,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import DatasetMapperTTA, GeneralizedRCNNWithTTA, build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
    get_event_storage,
)


def get_evaluator(cfg, dataset_name, output_folder):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


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

    logger = logging.getLogger("detectron2.trainer")

    if cfg.TEST.PRECISE_BN.ENABLED and len(get_bn_modules(model)) > 0:
        with EventStorage():  # capture events in a new storage to discard them
            train_data = build_detection_train_loader(cfg)
            logger.info(
                "Running precise-BN for {} iterations... ".format(cfg.TEST.PRECISE_BN.NUM_ITER)
                + "Note that this could produce different statistics every time."
            )
            update_bn_stats(model, train_data, cfg.TEST.PRECISE_BN.NUM_ITER)

    with inference_context(model):
        results = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:
            if cfg.OUTPUT_DIR:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            else:
                output_folder = None

            # NOTE: creating evaluator after dataset is loaded as there might be dependency.
            data_loader = build_detection_test_loader(cfg, dataset_name)
            evaluator = get_evaluator(cfg, dataset_name, output_folder)
            results_per_dataset = inference_on_dataset(model, data_loader, evaluator)
            if comm.is_main_process():
                results[dataset_name] = results_per_dataset
                if is_final:
                    print_csv_format(results_per_dataset)

            if is_final and cfg.TEST.AUG_ON:  # TODO make this logic simpler
                # In the end of training, run an evaluation with TTA
                if cfg.OUTPUT_DIR:
                    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_TTA", dataset_name)
                else:
                    output_folder = None

                newcfg = cfg.clone()
                newcfg.defrost()
                newcfg.INPUT.MIN_SIZE_TEST = 0  # disable resizing
                logger.info("Running inference with test-time augmentation ...")
                data_loader = build_detection_test_loader(
                    cfg, dataset_name, mapper=DatasetMapper(newcfg, is_train=False)
                )
                model = GeneralizedRCNNWithTTA(cfg, model, DatasetMapperTTA(cfg))
                evaluator = get_evaluator(cfg, dataset_name, output_folder)
                results_per_dataset = inference_on_dataset(model, data_loader, evaluator)
                logger.info(
                    "Evaluation results on {} with test-time augmentation:".format(dataset_name)
                )
                if comm.is_main_process():
                    print_csv_format(results_per_dataset)

    if is_final and cfg.TEST.EXPECTED_RESULTS and comm.is_main_process():
        assert len(results) == 1, "Results verification only supports one dataset!"
        verify_results(cfg, results[cfg.DATASETS.TEST[0]])

    try:
        storage = get_event_storage()
    except Exception:  # do_test may be called outside training
        pass
    else:
        storage.put_scalars(**flatten_results_dict(results), smoothing_hint=False)

    return results


def create_eval_hook(cfg, model):
    def after_step_callback(trainer):
        is_final = trainer.iter + 1 == trainer.max_iter
        if is_final or (
            cfg.TEST.EVAL_PERIOD > 0 and (trainer.iter + 1) % cfg.TEST.EVAL_PERIOD == 0
        ):
            do_test(cfg, model, is_final=is_final)
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

    data_loader = build_detection_train_loader(cfg, start_iter=start_iter)

    """
    Here we use a pre-defined training loop (the trainer) with hooks to run the training.
    This makes it easier to reuse existing utilities.
    If you'd like to do anything fancier than the standard training loop,
    consider writing your own loop or subclassing the trainer.
    """
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
        # Note: when precise BN is enabled, some checkpoints will have more precise
        # statistics than others, if they are saved immediately after eval.
        trainer_hooks.append(hooks.PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

    trainer.register_hooks(trainer_hooks)
    trainer.train(start_iter, max_iter)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # Enable hacky research that uses global config. You usually don't need it
    # set_global_cfg(cfg.GLOBAL)
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

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

# Set up custom environment before anything else is imported
from detectron2.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import datetime
import logging
import os
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg, set_global_cfg
from detectron2.data import (
    DetectionTransform,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import SimpleTrainer, hooks, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    inference_context,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import DetectionTransformTTA, GeneralizedRCNNWithTTA, build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.events import JSONWriter, TensorboardXWriter, get_event_storage
from detectron2.utils.logger import setup_logger


class MetricPrinter:
    def __init__(self, max_iter):
        self.logger = logging.getLogger("detectron2.trainer")
        self._max_iter = max_iter

    def write(self):
        storage = get_event_storage()
        iteration = storage.iteration

        data_time, time = None, None
        eta_string = "N/A"
        try:
            data_time = storage.history("data_time").median(20)
            time = storage.history("time").global_avg()
            eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:  # they may not exist in the first few iterations (due to warmup)
            pass

        # NOTE: max mem is parsed by grep
        self.logger.info(
            """\
eta: {eta}  iter: {iter}  {losses}  \
{time}  {data_time}  \
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
                time="time: {:.4f}".format(time) if time is not None else "",
                data_time="data_time: {:.4f}".format(data_time) if data_time is not None else "",
                lr=storage.history("lr").latest(),
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
            )
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
    if evaluator_type in ["semantic_seg", "coco_panoptic_seg"]:
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
        # TODO add per-machine primitives (https://github.com/fairinternal/detectron2/issues/138)
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "COCOPanopticEvaluator currently do not work with multiple machines."
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesEvaluator(dataset_name)
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
    if isinstance(model, DistributedDataParallel):
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    if not cfg.MODEL.LOAD_PROPOSALS:
        proposal_files_test = (None,) * len(cfg.DATASETS.TEST)
    else:
        proposal_files_test = cfg.DATASETS.PROPOSAL_FILES_TEST
    assert len(proposal_files_test) == len(cfg.DATASETS.TEST)
    logger = logging.getLogger("detectron2.trainer")

    with inference_context(model):
        results = []
        for dataset_name, proposal_file in zip(cfg.DATASETS.TEST, proposal_files_test):
            if cfg.OUTPUT_DIR:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
                if comm.is_main_process():
                    os.makedirs(output_folder, exist_ok=True)
                comm.synchronize()
            else:
                output_folder = None

            # NOTE: creating evaluator after dataset is loaded as there might be dependency.
            data_loader = build_detection_test_loader(cfg, dataset_name, proposal_file)
            evaluator = get_evaluator(cfg, dataset_name, output_folder)
            results_per_dataset = inference_on_dataset(model, data_loader, evaluator)
            if comm.is_main_process():
                results.append(results_per_dataset)
                if is_final:
                    print_csv_format(results_per_dataset)

            if is_final and cfg.TEST.AUG_ON:
                # In the end of training, run an evaluation with TTA
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_TTA", dataset_name)
                if comm.is_main_process():
                    os.makedirs(output_folder, exist_ok=True)
                comm.synchronize()
                assert proposal_file is None, "TTA with pre-computed proposal is not supported now."

                newcfg = cfg.clone()
                newcfg.defrost()
                newcfg.INPUT.MIN_SIZE_TEST = 0
                data_loader = build_detection_test_loader(
                    cfg, dataset_name, transform=DetectionTransform(newcfg, is_train=False)
                )
                transform = DetectionTransformTTA(cfg)
                model = GeneralizedRCNNWithTTA(cfg, model, transform)
                evaluator = get_evaluator(cfg, dataset_name, output_folder)
                results_per_dataset = inference_on_dataset(model, data_loader, evaluator)
                logger.info(
                    "Evaluation results on {} with test-time augmentation:".format(dataset_name)
                )
                if comm.is_main_process():
                    print_csv_format(results_per_dataset)

    if is_final and cfg.TEST.EXPECTED_RESULTS and comm.is_main_process():
        assert len(results) == 1, "Results verification only supports one dataset!"
        verify_results(cfg, results[0])
    return results


def create_after_step_hook(cfg, model, optimizer, scheduler, periodic_checkpointer):
    """
    Create a hook that performs some pre-defined tasks used in this script
    (evaluation, LR scheduling, checkpointing).
    """
    cfg_str = cfg.dump()

    def after_step_callback(trainer):
        if (
            cfg.TEST.EVAL_PERIOD > 0
            and (trainer.iter + 1) % cfg.TEST.EVAL_PERIOD == 0
            and trainer.iter != trainer.max_iter - 1
        ):
            results = do_test(cfg, model, is_final=False)

            for dataset_name, results_per_dataset in zip(cfg.DATASETS.TEST, results):
                for task, metrics_per_task in results_per_dataset.items():
                    for metric, value in metrics_per_task.items():
                        key = "{}/{}/{}".format(dataset_name, task, metric)
                        trainer.storage.put_scalar(key, value, smoothing_hint=False)
            # Evaluation may take different time among workers.
            # A barrier make them start the next iteration together.
            comm.synchronize()

        trainer.storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
        scheduler.step()
        periodic_checkpointer.step(trainer.iter, cfg=cfg_str)

    return hooks.CallbackHook(after_step=after_step_callback)


def do_train(cfg, model):
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(model, optimizer, scheduler, cfg.OUTPUT_DIR)
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHT).get("iteration", 0)
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

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
        create_after_step_hook(cfg, model, optimizer, scheduler, periodic_checkpointer),
    ]
    if comm.is_main_process():
        writers = [
            MetricPrinter(max_iter),
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
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg.GLOBAL)

    colorful_logging = not args.no_color
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    comm.synchronize()

    logger = setup_logger(output_dir, color=colorful_logging, distributed_rank=comm.get_rank())
    logger.info(
        "Using {} GPUs per machine. Rank of current process: {}".format(
            args.num_gpus, comm.get_rank()
        )
    )
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info(
        "Loaded config file {}:\n{}".format(args.config_file, open(args.config_file, "r").read())
    )
    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        path = os.path.join(output_dir, "config.yaml")
        with open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    output_dir = cfg.OUTPUT_DIR

    if args.eval_only:
        checkpointer = DetectionCheckpointer(model, save_dir=output_dir)
        checkpointer.load(cfg.MODEL.WEIGHT)
        return do_test(cfg, model)

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


def parse_args(in_args=None):
    """
    Method optionally supports passing arguments. If not provided it is read in
    from sys.argv. If providing it should be a list as per python documentation.
    See https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args
    """
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
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

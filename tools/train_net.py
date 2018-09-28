r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from torch_detectron.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from torch_detectron.config import cfg
from torch_detectron.config.data import make_data_loader
from torch_detectron.config.solver import make_lr_scheduler
from torch_detectron.config.solver import make_optimizer
from torch_detectron.engine.inference import inference
from torch_detectron.engine.logger import setup_logger
from torch_detectron.engine.trainer import do_train
from torch_detectron.modeling.model_builder import build_detection_model
from torch_detectron.utils.checkpoint import DetectronCheckpointer
from torch_detectron.utils.collect_env import collect_env_info
from torch_detectron.utils.imports import import_file
from torch_detectron.utils.miscellaneous import mkdir


def train(cfg, local_rank, distributed):
    data_loader = make_data_loader(cfg, is_train=True, is_distributed=distributed)

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    max_iter = cfg.SOLVER.MAX_ITER

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = local_rank == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        max_iter,
        device,
        distributed,
        arguments,
    )

    return model


def test(cfg, model, distributed):
    data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    inference(
        model,
        data_loader_val,
        iou_types=iou_types,
        box_only=cfg.MODEL.RPN_ONLY,
        device=cfg.MODEL.DEVICE,
        expected_results=cfg.TEST.EXPECTED_RESULTS,
        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
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

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("torch_detectron", output_dir, args.local_rank)
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()

r"""
Basic training script for PyTorch

Run with

python -m torch_detectron.train --config-file path_to_config_file

to perform training
"""
import argparse
import logging
import os

import torch

from torch_detectron.core.inference import inference
from torch_detectron.core.trainer import do_train
from torch_detectron.helpers.config_utils import load_config
from torch_detectron.helpers.config_utils import update_config_with_args
from torch_detectron.utils.checkpoint import Checkpoint
from torch_detectron.utils.logging import setup_logger
from torch_detectron.utils.miscellaneous import mkdir


def train(config, local_rank, distributed):
    # FIXME this is not great
    config.TRAIN.DATA.DATALOADER.SAMPLER.DISTRIBUTED = distributed
    data_loader = config.TRAIN.DATA()

    model = config.MODEL()
    device = config.DEVICE
    model.to(device)

    optimizer = config.SOLVER.OPTIM(model)
    scheduler = config.SOLVER.SCHEDULER(optimizer)
    max_iter = config.SOLVER.MAX_ITER

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    arguments = {}
    arguments["iteration"] = 0

    save_dir = config.SAVE_DIR
    checkpoint_file = config.CHECKPOINT

    checkpointer = Checkpoint(model, optimizer, scheduler, save_dir, local_rank)

    if checkpoint_file:
        extra_checkpoint_data = checkpointer.load(checkpoint_file)
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


def test(config, model, distributed):
    config.TEST.DATA.DATALOADER.SAMPLER.DISTRIBUTED = distributed
    if distributed:
        # TODO distributed sampler doesn't support shuffle=False
        config.TEST.DATA.DATALOADER.SAMPLER.SHUFFLE = True
    data_loader_val = config.TEST.DATA()
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if config.MODEL.USE_MASK:
        iou_types = iou_types + ("segm",)
    inference(
        model,
        data_loader_val,
        iou_types=iou_types,
        box_only=config.MODEL.RPN_ONLY,
        expected_results=config.TEST.EXPECTED_RESULTS,
        expected_results_sigma_tol=config.TEST.EXPECTED_RESULTS_SIGMA_TOL,
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch/configs/rpn_r50.py",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument("--local_rank", type=int, default=0)
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

    config = load_config(args.config_file)
    update_config_with_args(config, args.opts)

    save_dir = config.SAVE_DIR
    if save_dir:
        mkdir(save_dir)

    setup_logger("torch_detectron", save_dir, args.local_rank)

    logger = logging.getLogger("torch_detectron")
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(config))

    model = train(config, args.local_rank, args.distributed)

    if config.DO_TEST:
        test(config, model, args.distributed)


if __name__ == "__main__":
    main()

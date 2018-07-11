r"""
Basic training script for PyTorch

Run with

python -m torch_detectron.train --config-file path_to_config_file

to perform training
"""
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


def train(config):
    use_distributed = config.distributed
    local_rank = config.local_rank

    # FIXME this is not great
    config.config.TRAIN.DATA.DATALOADER.SAMPLER.DISTRIBUTED = use_distributed
    data_loader = config.config.TRAIN.DATA()

    model = config.config.MODEL()
    device = config.config.DEVICE
    model.to(device)

    optimizer = config.config.SOLVER.OPTIM(model)
    scheduler = config.config.SOLVER.SCHEDULER(optimizer)
    max_iter = config.config.SOLVER.MAX_ITER

    if use_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    arguments = {}
    arguments["iteration"] = 0

    save_dir = config.config.SAVE_DIR
    checkpoint_file = config.config.CHECKPOINT

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
        use_distributed,
        arguments,
    )

    return model


def test(config, model):
    use_distributed = config.distributed
    config.config.TEST.DATA.DATALOADER.SAMPLER.DISTRIBUTED = False
    data_loader_val = config.config.TEST.DATA()
    if use_distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if config.config.MODEL.USE_MASK:
        iou_types = iou_types + ("segm",)
    inference(
        model,
        data_loader_val,
        iou_types=iou_types,
        box_only=config.config.MODEL.RPN_ONLY,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch/configs/rpn_r50.py",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    config = load_config(args.config_file)
    config.local_rank = args.local_rank
    config.distributed = args.distributed
    update_config_with_args(config.config, args.opts)

    save_dir = config.config.SAVE_DIR
    if save_dir:
        mkdir(save_dir)

    setup_logger("torch_detectron", save_dir, args.local_rank)

    logger = logging.getLogger("torch_detectron")
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info({k: v for k, v in config.__dict__.items() if not k.startswith("__")})

    model = train(config)

    if args.local_rank == 0 and config.config.DO_TEST:
        test(config, model)


if __name__ == "__main__":
    main()

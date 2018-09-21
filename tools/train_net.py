r"""
Basic training script for PyTorch
"""
import argparse
import logging
import os

import torch

from torch_detectron.config import cfg

from torch_detectron.engine.inference import inference
from torch_detectron.engine.trainer import do_train
from torch_detectron.config.data import make_data_loader
from torch_detectron.config.solver import make_optimizer
from torch_detectron.config.solver import make_lr_scheduler
from torch_detectron.config.utils import import_file
from torch_detectron.modeling.model_builder import build_detection_model
from torch_detectron.utils.checkpoint import Checkpoint
from torch_detectron.engine.logger import setup_logger
from torch_detectron.utils.miscellaneous import mkdir


# TODO handle model retraining
def load_from_pretrained_checkpoint(cfg, model):
    if not cfg.MODEL.WEIGHT:
        return
    weight_path = cfg.MODEL.WEIGHT
    if weight_path.startswith("catalog://"):
        paths_catalog = import_file("torch_detectron.config.paths_catalog", cfg.PATHS_CATALOG, True)
        ModelCatalog = paths_catalog.ModelCatalog
        weight_path = ModelCatalog.get(weight_path[len("catalog://"):])

    if weight_path.endswith("pkl"):
        from torch_detectron.utils.c2_model_loading import _load_c2_pickled_weights, _rename_weights_for_R50
        state_dict = _load_c2_pickled_weights(weight_path)
        state_dict = _rename_weights_for_R50(state_dict)
    else:
        state_dict = torch.load(weight_path)
    if cfg.MODEL.RPN.USE_FPN or cfg.MODEL.ROI_HEADS.USE_FPN:
        model.backbone[0].stem.load_state_dict(state_dict, strict=False)
        model.backbone[0].load_state_dict(state_dict, strict=False)
    else:
        model.backbone.stem.load_state_dict(state_dict, strict=False)
        model.backbone.load_state_dict(state_dict, strict=False)

        model.roi_heads.heads[0].feature_extractor.head.load_state_dict(
            state_dict, strict=False
        )


def train(cfg, local_rank, distributed):
    data_loader = make_data_loader(cfg, is_train=True, is_distributed=distributed)

    model = build_detection_model(cfg)
    load_from_pretrained_checkpoint(cfg, model)
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
    checkpoint_file = cfg.CHECKPOINT

    checkpointer = Checkpoint(model, optimizer, scheduler, output_dir, local_rank)

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
        default="/private/home/fmassa/github/detectron.pytorch/configs/rpn_r50.py",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        '--skip-test',
        dest='skip_test',
        help='Do not test the final model',
        action='store_true'
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

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from maskrcnn_benchmark.detection import DetectionCheckpointer
from maskrcnn_benchmark.detection import build_detection_model
from maskrcnn_benchmark.detection import coco_evaluation
from maskrcnn_benchmark.detection import get_cfg
from maskrcnn_benchmark.detection import make_detection_data_loader
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:12457")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = args.num_gpus

    if num_gpus > 1:
        mp.spawn(main_worker, nprocs=num_gpus, args=(args,), daemon=False)
    else:
        main_worker(0, args)


def main_worker(worker_id, args):
    if args.num_gpus > 1:
        dist.init_process_group(
            backend="NCCL", init_method=args.dist_url, world_size=args.num_gpus, rank=worker_id
        )
        torch.cuda.set_device(worker_id)

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger(save_dir, get_rank())
    logger.info("Using {} GPUs".format(args.num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectionCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

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
        cfg, is_train=False, is_distributed=args.num_gpus > 1
    )
    for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
        coco_evaluation(
            model,
            data_loader_val,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            output_folder=output_folder,
        )
        synchronize()


if __name__ == "__main__":
    main()

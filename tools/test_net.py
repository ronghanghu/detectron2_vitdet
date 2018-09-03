import argparse
import os

from torch_detectron.utils.c2_model_loading import load_c2_weights_faster_rcnn_resnet50_c4
from torch_detectron.modeling.config import cfg
from torch_detectron.engine.inference import inference
from torch_detectron.engine.logger import setup_logger
from torch_detectron.modeling.model_builder import build_detection_model

import torch


def make_model(cfg):

    load_weights_faster_rcnn_resnet50_c4(model, "detectron.pytorch/torch_detectron/core/models/faster_rcnn_resnet50.pth")
    return model


def make_transform(cfg):
    from torch_detectron.utils import data_transforms as T
    resize_transform = T.Resize(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN)

    to_bgr255 = True
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255)

    flip_prob = 0  # TODO need to handle different train / test values
    transform = T.Compose(
        [
            resize_transform,
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform

def make_coco_dataset(cfg):
    from torch_detectron.datasets.coco import COCODataset
    dataset = COCODataset(
            "/private/home/fmassa/github/detectron.pytorch/configs/datasets/coco/annotations/instances_minival2014.json",
            "/private/home/fmassa/github/detectron.pytorch/configs/datasets/coco/val2014",
            remove_images_without_annotations=False,
            transforms=make_transform(cfg))
    return dataset


def make_data_sampler(dataset, shuffle, distributed):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        assert distributed == False, "Distributed with no shuffling on the dataset not supported"
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = sorted(bins.copy())
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_batch):
    from torch_detectron.utils.data_samplers import GroupedBatchSampler
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    return batch_sampler


def make_data_loader_val(cfg):
    from torch_detectron.utils.data_collate import BatchCollator
    import torch.utils.data

    collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
    dataset = make_coco_dataset(cfg)
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collator, shuffle=False)
    return data_loader




def load_from_c2(model, weights_file):
    pass

def load_from_checkpoint(model, checkpoint):
    pass


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--checkpoint", default="", metavar="FILE", help="path to checkpoint file"
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

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("torch_detectron", save_dir, args.local_rank)
    logger.info(cfg)
    
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    load_c2_weights_faster_rcnn_resnet50_c4(model, "/private/home/fmassa/model_final_faster_rcnn.pkl")

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        # TODO find a better way of serializing the weights
        # that avoids this ugly workaround
        model = torch.nn.DataParallel(model)
        load_state_dict(model, checkpoint["model"])
        model = model.module

    data_loader_val = make_data_loader_val(cfg)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    inference(
        model,
        data_loader_val,
        iou_types=iou_types,
        box_only=cfg.MODEL.RPN_ONLY,
    )


if __name__ == "__main__":
    main()

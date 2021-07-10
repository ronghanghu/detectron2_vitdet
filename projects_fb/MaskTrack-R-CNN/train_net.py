#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Main training script for MaskTrackRCNN.

This scripts reads a given config file and runs the training or evaluation.
"""

import copy
import logging
import os
import torch

import detectron2.projects_fb.masktrack_rcnn.lib.datasets.builtin_ytvis  # noqa
import detectron2.projects_fb.masktrack_rcnn.lib.mrcnn.masktrack_r_cnn  # noqa
import detectron2.projects_fb.masktrack_rcnn.lib.mrcnn.masktrack_roi_heads  # noqa
import detectron2.projects_fb.masktrack_rcnn.lib.mrcnn.track_head  # noqa
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.build import build_batch_data_loader, trivial_batch_collator
from detectron2.data.common import MapDataset
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.projects_fb.masktrack_rcnn.lib.datasets.dataloader_utils import (
    get_VIS_dataset_dicts,
)
from detectron2.projects_fb.masktrack_rcnn.lib.datasets.dataset_mapper import YTVISDatasetMapper
from detectron2.projects_fb.masktrack_rcnn.lib.mrcnn.config import add_masktrack_r_cnn_config
from detectron2.projects_fb.masktrack_rcnn.lib.utils.ytvis_evaluator import YTVISEvaluator
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for the YTVIS dataset.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return YTVISEvaluator(dataset_name, True, output_folder)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        )
        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

    @classmethod
    def build_train_loader(cls, cfg):
        dataset = get_VIS_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))
        mapper = YTVISDatasetMapper(cfg, is_train=True)
        dataset = MapDataset(dataset, mapper)
        data_loader = build_batch_data_loader(
            dataset,
            sampler,
            total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
            aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )
        return data_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset = get_VIS_dataset_dicts(
            dataset_name,
            filter_empty=False,
            proposal_files=[
                cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
            ]
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )

        # load one frame at a time for inference
        new_dataset = []
        for vid_dict in dataset:
            for frame in vid_dict["frames"]:
                new_vid_dict = copy.deepcopy(vid_dict)
                new_vid_dict["frames"] = [frame]
                new_dataset.append(new_vid_dict)

        mapper = YTVISDatasetMapper(cfg, is_train=False)
        new_dataset = MapDataset(new_dataset, mapper)
        sampler = InferenceSampler(len(new_dataset))
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
        data_loader = torch.utils.data.DataLoader(
            new_dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
        )
        return data_loader


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_masktrack_r_cnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


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

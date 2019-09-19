import os
import sys
import glob
import pickle
import argparse
import logging
from typing import ClassVar, Dict, Any, List
import torch

from detectron2.config import get_cfg, CfgNode
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.common import MapDataset
from detectron2.data.build import trivial_batch_collator
from detectron2.data import samplers, DatasetFromList
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from densepose.structures import DensePoseDataRelative
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose import (
    DensePoseResultsSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)


DOC = """Apply Net - a tool to print / visualize DensePose results
"""

LOGGER_NAME = 'apply_net'
logger = logging.getLogger(LOGGER_NAME)

_ACTION_REGISTRY: Dict[str, 'Action'] = {}


class Action(object):

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-v", "--verbosity", action="count",
            help="Verbose mode. Multiple -v options increase the verbosity.")


def register_action(cls: type):
    """
    Decorator for action classes to automate action registration
    """
    global _ACTION_REGISTRY
    _ACTION_REGISTRY[cls.COMMAND] = cls
    return cls


class InferenceAction(Action):

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(InferenceAction, cls).add_arguments(parser)
        parser.add_argument(
            "cfg", metavar="<config>", help="Config file")
        parser.add_argument(
            "model", metavar="<model>", help="Model file")
        parser.add_argument(
            "input", metavar="<input>", help="Input data")

    @classmethod
    def execute(cls: type, args: argparse.Namespace):
        logger.info(f"Loading config from {args.cfg}")
        cfg = cls._setup_config(args.cfg)
        logger.info(f"Loading model from {args.model}")
        model = cls._load_model(cfg, args.model)
        logger.info(f"Loading data from {args.input}")
        dataset = cls._setup_dataset(args.input)
        if len(dataset) == 0:
            logger.warning(f"No input images for {args.input}")
            return
        data_loader = cls._setup_data_loader(cfg, dataset)
        model.eval()
        context = cls.create_context(args)
        for _idx, batch in enumerate(data_loader):
            with torch.no_grad():
                outputs = model(batch)
                cls.execute_on_outputs(context, batch, outputs)
        cls.postexecute(context)

    @classmethod
    def _setup_config(cls: type, config_fpath: str):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.freeze()
        return cfg

    @classmethod
    def _load_model(cls: type, cfg: CfgNode, model_fpath: str):
        model = build_model(cfg)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(model_fpath)
        return model

    @classmethod
    def _setup_dataset(cls: type, input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [
                fname for fname in os.listdir(input_spec)
                if os.path.isfile(os.path.join(input_spec, fname))
            ]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return DatasetFromList([{"file_name": fname} for fname in file_list])

    @classmethod
    def _setup_data_loader(cls: type, cfg: CfgNode, dataset):
        dataset_mapper = DatasetMapper(cfg, False)
        dataset_mapped = MapDataset(dataset, dataset_mapper)
        image_sampler = samplers.InferenceSampler(len(dataset_mapped))
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            image_sampler, 1, drop_last=False
        )
        data_loader = torch.utils.data.DataLoader(
            dataset_mapped,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
        )
        return data_loader


@register_action
class DumpAction(InferenceAction):
    """
    Dump action that outputs results to a pickle file
    """

    COMMAND: ClassVar[str] = "dump"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(
            cls.COMMAND, help='Dump model outputs to a file.')
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(DumpAction, cls).add_arguments(parser)
        parser.add_argument(
            "--output",
            metavar="<dump_file>",
            default='results.pkl',
            help="File name to save dump to")

    @classmethod
    def execute_on_outputs(
            cls: type,
            context: Dict[str, Any],
            batch: List[dict],
            outputs: List[Instances]):
        for entry, instances in zip(batch, outputs):
            image_fpath = entry['file_name']
            logger.info(f'Processing {image_fpath}')
            entry['instances'] = instances
            context['results'].append(entry)

    @classmethod
    def create_context(cls: type, args: argparse.Namespace):
        context = {
            'results': [],
            'out_fname': args.output,
        }
        return context

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        out_fname = context['out_fname']
        with open(out_fname, 'wb') as hFile:
            pickle.dump(context['results'], hFile)
            logger.info(f'Output saved to {out_fname}')


@register_action
class ShowAction(InferenceAction):
    """
    Show action that visualizes selected entries on an image
    """

    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        'dp_segm': DensePoseResultsSegmentationVisualizer(),
        'dp_u': DensePoseResultsUVisualizer(),
        'dp_v': DensePoseResultsVVisualizer(),
        'bbox': ScoredBoundingBoxVisualizer(min_score=0.8),
    }

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(
            cls.COMMAND, help='Visualize selected entries')
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(ShowAction, cls).add_arguments(parser)
        parser.add_argument(
            'visualizations',
            metavar='<visualizations>',
            help="Comma separated list of visualizations, possible values: "
                 "[{}]".format(','.join(sorted(cls.VISUALIZERS.keys()))))
        parser.add_argument(
            "--output",
            metavar="<image_file>",
            default='outputres.png',
            help="File name to save output to")

    @classmethod
    def execute_on_outputs(
            cls: type,
            context: Dict[str, Any],
            batch: List[dict],
            outputs: List[Instances]):
        import cv2
        import numpy as np
        visualizer = context['visualizer']
        for entry, output in zip(batch, outputs):
            image_fpath = entry['file_name']
            logger.info(f'Processing {image_fpath}')
            image = cv2.imread(image_fpath, cv2.IMREAD_GRAYSCALE)
            h = entry['image'].size(1)
            w = entry['image'].size(2)
            image = cv2.resize(image, (w, h), cv2.INTER_LINEAR)
            image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
            if not output['instances'].has('pred_densepose'):
                continue
            datas = cls._extract_data_for_visualizers(
                context['vis_specs'], output['instances'])
            image_vis = visualizer.visualize(image, datas)
            entry_idx = context['entry_idx'] + 1
            out_fname = cls._get_out_fname(entry_idx, context['out_fname'])
            cv2.imwrite(out_fname, image_vis)
            logger.info(f'Output saved to {out_fname}')
            context['entry_idx'] += 1

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        pass

    @classmethod
    def _get_out_fname(cls: type, entry_idx: int, fname_base: str):
        base, ext = os.path.splitext(fname_base)
        return base + '.{0:04d}'.format(entry_idx) + ext

    @classmethod
    def create_context(cls: type, args: argparse.Namespace) -> Dict[str, Any]:
        vis_specs = args.visualizations.split(',')
        visualizers = []
        for vis_spec in vis_specs:
            vis = cls.VISUALIZERS[vis_spec]
            visualizers.append(vis)
        context = {
            'vis_specs': vis_specs,
            'visualizer': CompoundVisualizer(visualizers),
            'out_fname': args.output,
            'entry_idx': 0,
        }
        return context

    @classmethod
    def _extract_data_for_visualizers(
            cls: type, vis_specs: List[str], instances: Instances):
        boxes_xywh = instances.pred_boxes.tensor
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]
        scores = instances.scores
        dp_results = instances.pred_densepose.to_result(boxes_xywh)
        datas = []
        for vis_spec in vis_specs:
            datas.append((boxes_xywh, scores) if 'bbox' == vis_spec else dp_results)
        return datas

    @classmethod
    def _extract_data_for_visualizers_from_entry(
            cls: type, vis_specs: List[str], entry: Dict[str, Any]):
        dp_list = []
        bbox_list = []
        for annotation in entry['annotations']:
            is_valid, _ = DensePoseDataRelative.validate_annotation(annotation)
            if not is_valid:
                continue
            bbox = torch.as_tensor(annotation['bbox'])
            bbox_list.append(bbox)
            dp_data = DensePoseDataRelative(annotation)
            dp_list.append(dp_data)
        datas = []
        for vis_spec in vis_specs:
            datas.append(bbox_list if 'bbox' == vis_spec else (bbox_list, dp_list))
        return datas


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DOC,
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=120))
    parser.set_defaults(func=lambda _: parser.print_help(sys.stdout))
    subparsers = parser.add_subparsers(title='Actions')
    for _, action in _ACTION_REGISTRY.items():
        action.add_parser(subparsers)
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    verbosity = args.verbosity if hasattr(args, 'verbosity') else None
    global logger
    logger = setup_logger(name=LOGGER_NAME)
    logger.setLevel(verbosity_to_level(verbosity))
    args.func(args)


if __name__ == "__main__":
    main()

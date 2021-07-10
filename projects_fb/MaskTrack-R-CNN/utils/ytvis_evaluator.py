# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import collections
import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
import torch

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.file_io import PathManager

from pycocotools_ytvos.ytvos import YTVOS
from pycocotools_ytvos.ytvoseval import YTVOSeval


def results2json_videoseg(vid_frame_ids, outputs):
    results = []
    vid_objs = {}
    for idx in range(len(vid_frame_ids)):
        # assume results is ordered
        vid_id, frame_id = vid_frame_ids[idx]
        if idx == len(vid_frame_ids) - 1:
            is_last = True
        else:
            _, frame_id_next = vid_frame_ids[idx + 1]
            is_last = frame_id_next == 0
        det, seg = outputs[idx]
        for obj_id in det:
            segm = seg[obj_id]
            label = det[obj_id]["label"]
            score = det[obj_id]["score"]
            if obj_id not in vid_objs:
                vid_objs[obj_id] = {"scores": [], "cats": [], "segms": {}}
            vid_objs[obj_id]["scores"].append(score)
            vid_objs[obj_id]["cats"].append(label)
            segm["counts"] = segm["counts"].decode()
            vid_objs[obj_id]["segms"][frame_id] = segm

        if is_last:
            # store results of  the current video
            for _obj_id, obj in vid_objs.items():
                data = dict()

                data["video_id"] = vid_id
                data["score"] = np.array(obj["scores"]).mean().item()
                # majority voting for sequence category
                # "+1" below is hacky reverse id_map
                data["category_id"] = np.bincount(np.array(obj["cats"])).argmax().item() + 1

                vid_seg = []
                for fid in range(frame_id + 1):
                    if fid in obj["segms"]:
                        vid_seg.append(obj["segms"][fid])
                    else:
                        vid_seg.append(None)
                data["segmentations"] = vid_seg
                results.append(data)
            vid_objs = {}
    return results


def ytvos_eval(result_file, result_types, ytvos, max_dets=(100, 300, 1000)):
    assert isinstance(ytvos, YTVOS)

    ytvos_dets = ytvos.loadRes(result_file)
    vid_ids = ytvos.getVidIds()
    for res_type in result_types:
        iou_type = res_type
        ytvosEval = YTVOSeval(ytvos, ytvos_dets, iou_type)
        ytvosEval.params.vidIds = vid_ids
        if res_type == "proposal":
            ytvosEval.params.useCats = 0
            ytvosEval.params.maxDets = list(max_dets)
        ytvosEval.evaluate()
        ytvosEval.accumulate()
        ytvosEval.summarize()


class YTVISEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation for
    YTVIS dataset.
    """

    def __init__(self, dataset_name, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:

                    "json_file": the path to the annotation file

            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "results.json" a json file of results.
        """
        from pycocotools_ytvos.ytvos import YTVOS

        self._tasks = ("bbox", "segm")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        assert hasattr(self._metadata, "json_file")

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._ytvos_api = YTVOS(json_file)

        # Val/test set json files do not contain annotations (evaluation must be
        # performed using the YTVIS evaluation server).
        self._do_evaluation = "annotations" in self._ytvos_api.dataset
        self._save_vis = False

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a MaskTrackRCNN model.
            outputs: the outputs of a MaskTrackRCNN model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """

        for input, output in zip(inputs, outputs):
            prediction = {
                "frame_idx": input["frame_idx"],
                "video_id": input["video_id"],
                "instances": output["instances"].to(self._cpu_device),
            }

            if self._save_vis:
                prediction.update(
                    {
                        # Needed to visualize predictions
                        "is_first": input["is_first"],
                        "file_name": input["file_name"],
                    }
                )

            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[YTVISEvaluator] Did not receive valid predictions.")
            return {}

        # prepare inputs for results2json_videoseg
        vid_frame_ids = [(pred["video_id"], pred["frame_idx"]) for pred in predictions]
        outputs = []
        for pred in predictions:
            instances = pred["instances"]

            det = {}
            seg = {}
            for i in range(len(instances)):
                obj_id = instances.pred_obj_ids[i].item()
                label = instances.pred_classes[i].item()
                score = instances.scores[i].item()
                det[obj_id] = {"label": label, "score": score}
                mask = instances.pred_masks[i]
                seg[obj_id] = mask_util.encode(
                    np.array(mask[:, :, None], order="F", dtype="uint8")
                )[0]

            outputs.append((det, seg))

        pred_results = results2json_videoseg(vid_frame_ids, outputs)

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "results.json")
            with PathManager.open(file_path, "w") as f:
                json.dump(pred_results, f)
            file_path = PathManager.get_local_path(file_path)

            if self._do_evaluation:
                eval_types = ["segm"]
                eval_results = ytvos_eval(file_path, eval_types, self._ytvos_api)
                print(eval_results)

        if self._save_vis:
            from .ytvis_visualizer import YTVISVisualizer
            from PIL import Image

            visualizer = YTVISVisualizer(metadata=self._metadata)
            instance_colors = {}
            for pred in predictions:
                if pred["is_first"]:
                    instance_colors.clear()
                image_file = PathManager.get_local_path(pred["file_name"])
                image = np.array(Image.open(image_file))
                instances = pred["instances"]
                instances = instances[instances.scores > 0.5]
                vis = visualizer.draw_instance_predictions(
                    image, instances, instance_colors=instance_colors
                )
                vis_file = os.path.join(
                    self._output_dir, f"""vis/{pred["video_id"]:03}/{pred["frame_idx"]:02}.jpg"""
                )
                PathManager.mkdirs(os.path.dirname(vis_file))
                vis.save(vis_file)

        return collections.OrderedDict()

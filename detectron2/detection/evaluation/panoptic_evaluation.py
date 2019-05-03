import itertools
import json
import logging
import os
from collections import OrderedDict
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.inference import DatasetEvaluator

logger = logging.getLogger(__name__)


class COCOPanopticEvaluator(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`
    """

    def __init__(self, dataset_split, output_dir):
        """
        Args:
            dataset_split (str): name of the dataset split
            output_dir (str): output directory to save results for evaluation
        """
        self._dataset_split = dataset_split
        dataset_name = MetadataCatalog.get(dataset_split).dataset_name
        self._dataset_meta = MetadataCatalog.get(dataset_name)
        self._instances_contiguous_id_to_dataset_id = {
            v: k for k, v in self._dataset_meta.json_id_to_contiguous_id.items()
        }

        self._output_dir = output_dir
        self._predictions_dir = os.path.join(output_dir, "predictions")
        self._predictions_json = os.path.join(output_dir, "predictions.json")
        os.makedirs(self._predictions_dir, exist_ok=True)

    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._instances_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._dataset_meta.stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu()

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            Image.fromarray(id2rgb(panoptic_img.numpy())).save(
                os.path.join(self._predictions_dir, file_name_png)
            )
            segments_info = [self._convert_category_id(x) for x in segments_info]
            self._predictions.append(
                {
                    "image_id": input["image_id"],
                    "file_name": file_name_png,
                    "segments_info": segments_info,
                }
            )

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.all_gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))

        if not comm.is_main_process():
            return

        dataset_split_meta = MetadataCatalog.get(self._dataset_split)
        gt_json = dataset_split_meta.panoptic_json
        gt_folder = dataset_split_meta.panoptic_root

        with open(gt_json, "r") as f:
            json_data = json.load(f)
        json_data["annotations"] = self._predictions
        with open(self._predictions_json, "w") as f:
            json.dump(json_data, f)

        from panopticapi.evaluation import pq_compute

        pq_res = pq_compute(
            gt_json, self._predictions_json, gt_folder=gt_folder, pred_folder=self._predictions_dir
        )
        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        logger.info(results)
        return results
import glob
import logging
import os
import tempfile
from collections import OrderedDict
import torch
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.inference import DatasetEvaluator


class CityscapesEvaluator(DatasetEvaluator):
    """
    Evaluate instance segmentation results using cityscapes API.

    Note: does not work in distributed training for now
    """

    def __init__(self, dataset_split):
        """
        Args:
            dataset_split (str): the name of the dataset split
        """
        self._split_meta = MetadataCatalog.get(dataset_split)
        self._dataset_meta = MetadataCatalog.get(self._split_meta.dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_eval_")
        self._temp_dir = self._working_dir.name
        # All workers will write to the same results directory
        # TODO this does not work in distributed training
        self._temp_dir = comm.all_gather(self._temp_dir)[0]
        if self._temp_dir != self._working_dir.name:
            self._working_dir.cleanup()
        self._logger.info(
            "Writing cityscapes results to temporary directory {} ...".format(self._temp_dir)
        )

    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import name2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_txt = os.path.join(self._temp_dir, basename + "_pred.txt")

            output = output["detector"].to(self._cpu_device)
            num_instances = len(output)
            with open(pred_txt, "w") as fout:
                for i in range(num_instances):
                    pred_class = output.pred_classes[i]
                    class_name = self._dataset_meta.class_names[pred_class]
                    class_id = name2label[class_name].id
                    score = output.scores[i]
                    mask = output.pred_masks[i].numpy()
                    png_filename = os.path.join(
                        self._temp_dir, basename + "_{}_{}.png".format(i, class_name)
                    )

                    Image.fromarray(mask * 255).save(png_filename)
                    fout.write("{} {} {}\n".format(os.path.basename(png_filename), class_id, score))

    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        os.environ["CITYSCAPES_DATASET"] = os.path.abspath(
            os.path.join(self._split_meta.gt_dir, "..", "..")
        )
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluting
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = os.path.join(self._temp_dir, "gtInstances.json")

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
        groundTruthImgList = glob.glob(cityscapes_eval.args.groundTruthSearch)
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )["averages"]

        ret = OrderedDict()
        ret["segm"] = {"AP": results["allAp"], "AP50": results["allAp50%"]}
        self._working_dir.cleanup()
        return ret

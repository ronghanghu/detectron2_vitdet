from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
)

from newconfig import ConfigFile
from newconfig import LazyCall as L

train, test = ConfigFile.load_rel("./coco.py", ("train", "test"))
train.dataset.names = "coco_2017_train_panoptic_separated"
train.dataset.filter_empty = False
test.dataset.names = "coco_2017_val_panoptic_separated"


evaluator = L(DatasetEvaluators)(
    evaluators=[
        L(COCOEvaluator)(
            dataset_name="${....test.dataset.names}",
        ),
        L(SemSegEvaluator)(
            dataset_name="${....test.dataset.names}",
        ),
        L(COCOPanopticEvaluator)(
            dataset_name="${....test.dataset.names}",
        ),
    ]
)

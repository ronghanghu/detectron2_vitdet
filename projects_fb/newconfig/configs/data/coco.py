import detectron2.data.transforms as T
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

from newconfig import Config as D

train = D(
    build_detection_train_loader,
    dataset=D(get_detection_dataset_dicts, dataset_names=("coco_2017_val",)),
    mapper=D(
        DatasetMapper,
        is_train=True,
        augmentations=[
            D(
                T.ResizeShortestEdge,
                short_edge_length=(640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            D(T.RandomFlip, horizontal=True),
        ],
        image_format="BGR",
        use_instance_mask=True,
    ),
    total_batch_size=16,
    num_workers=4,
)

test = D(
    build_detection_test_loader,
    dataset=D(get_detection_dataset_dicts, dataset_names="coco_2017_val", filter_empty=False),
    mapper=D(
        DatasetMapper,
        is_train=False,
        augmentations=[
            D(T.ResizeShortestEdge, short_edge_length=800, max_size=1333),
        ],
        # ... a bit ugly
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

evaluator = D(
    COCOEvaluator,
    dataset_name="${..test.dataset.dataset_names}",
)

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.build import build_batch_data_loader
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation import LVISEvaluator

def build_data_loader(dataset_names, mapper, repeat_thresh_training_sampler, total_batch_size, num_workers):
    dataset = get_detection_dataset_dicts(
        names=dataset_names
    )

    repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
        dataset, repeat_thresh_training_sampler
    )
    sampler = RepeatFactorTrainingSampler(repeat_factors)

    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=True,
        num_workers=num_workers,
    )

dataloader = OmegaConf.create()

dataloader.train = L(build_data_loader)(
    dataset_names="lvis_v1_train",
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.RandomApply)(
                tfm_or_aug=L(T.AugmentationList)(
                    augs=[
                        L(T.ResizeShortestEdge)(
                            short_edge_length=[400, 500, 600], sample_style="choice"
                        ),
                        L(T.RandomCrop)(crop_type="absolute_range", crop_size=(384, 600)),
                    ]
                ),
                prob=0.5,
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            L(T.RandomFlip)(horizontal=True),
        ],
        image_format="RGB",
        use_instance_mask=True,
    ),
    repeat_thresh_training_sampler=0.001,
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="lvis_v1_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(LVISEvaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=300,
)

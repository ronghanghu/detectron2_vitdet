from .cityscapes import load_cityscapes_instances
from .coco import load_coco_json, load_sem_seg
from .register_coco import register_coco_instances, register_coco_panoptic_separated
from . import builtin  # this will register the builtin datasets

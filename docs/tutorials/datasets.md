# Use Custom Datasets

If you want to use a different dataset, but reuse dataloaders in detectron2,
you need to 

1. Register your dataset: tell detectron2 how to obtain your dataset.
2. Optionally, register some metadata for your dataset.

Next, we explain the above two concepts in details.


### Register a Dataset

To let detectron2 know how to obtain a dataset named "my_dataset", you need to:
```python
def get_dicts():
  ...
  return list[dict] in the following format

from detectron2.data import DatasetCatalog
DatasetCatalog.register("my_dataset", get_dicts)
```

Here, the snippet associates a dataset "my_dataset" with a function that returns the data.
If you do not modify downstream code (i.e., you use the standard data loader and data mapper),
then the function has to return a list of dicts in detectron2's standard dataset format.

For standard tasks
(instance detection, instance/semantic/panoptic segmentation, keypoint detection),
we use a format similar to COCO's json annotation
as the basic format to represent a dataset.

The format uses one dict to represent the annotations of
one image. The dict may have the following fields.
The fields are often optional, and some functions may be able to
infer certain fields from others if needed, e.g., the data loader
can read from "file_name" if "image" is not available.

+ `file_name`: the full path to the image file.
+ `sem_seg_file_name`: the full path to the ground truth semantic segmentation file.
+ `image`: the image in a numpy array.
+ `sem_seg`: semantic segmentation ground truth in a 2D numpy array. Values in the array represent
 		semantic labels.
+ `height`, `width`: integer. The size of image.
+ `image_id` (str): a string to identify this image. Mainly used during evaluation to identify the
		image. Each dataset may use it for different purposes.
+ `annotations` (list[dict]): the per-instance annotations of every
		instance in this image. Each annotation dict may contain:
	+ `bbox` (list[float]): list of 4 numbers representing the bounding box of the instance.
	+ `bbox_mode` (int): the format of bbox.
			It must be a member of [detectron2.structures.BoxMode](detectron2/structures/boxes.py).
		  Currently only supports `BoxMode.XYXY_ABS` and `BoxMode.XYWH_ABS`.
	+ `category_id` (int): an integer in the range [0, num_categories).
	+ `segmentation` (list[list[float]] or dict):
		+ For `list[list[float]]`, it represents the polygons of
			each object part. Each `list[float]` is one polygon in the
			format of `[x1, y1, ..., xn, yn]`.
			The Xs and Ys are either relative coordinates in [0, 1], or absolute coordinates,
			depend on whether "bbox_mode" is relative.
		+ For `dict`, it represents the per-pixel segmentation in COCO's RLE format.
	+ `keypoint`s (list[float]): in the format of [x1, y1, v1,..., xn, yn, vn].
		v[i] means the visibility of this keypoint.
		`n` must be equal to the number of keypoint categories.
		The Xs and Ys are either relative coordinates in [0, 1], or absolute coordinates,
		depend on whether "bbox_mode" is relative.

		Note that the coordinate annotations in COCO format are integers in range [0, H-1 or W-1].
		They are added by 0.5 to become absolute coordinates.
	+ `iscrowd`: 0 or 1. Whether this instance is labeled as COCO's "crowd region". Optional.
+ `proposal_boxes` (array): 2D numpy array with shape (K, 4). K precomputed proposal boxes for this image.
+ `proposal_objectness_logits` (array): numpy array with shape (K, ), which corresponds to the objectness
        logits of proposals in 'propopsal_boxes'.
+ `proposal_bbox_mode` (int): the format of the precomputed proposal bbox.
        It must be a member of [detectron2.structures.BoxMode](detectron2/structures/boxes.py).
        Default format is `BoxMode.XYXY_ABS`.


If your dataset is in COCO-format, you can also register it simply by
```python
from detectron2.data.datasts import register_coco_instances
register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")
```
which will take care of everything (including metadata) for you.


### "Metadata" for Datasets

Each dataset is associated with some metadata, accessible through
`MetadataCatalog.get(dataset_name).some_metadata`.
Metadata is a key-value mapping that contains primitive information that helps interpret what's in the dataset, e.g.,
names of classes, colors of classes, root of files, etc.
This information will be useful for augmentation, evaluation, visualization, logging, etc.
The structure of metadata depends on the what is needed from the corresponding downstream code.


If you register a new dataset through `DatasetCatalog.register`,
you may also want to add its corresponding metadata through
`MetadataCatalog.get(dataset_name).set(name, value)`, to enable the features that use these metadata.
You can do it like this:

```python
from detectron2.data import MetadataCatalog
MetadataCatalog.get("my_dataset").class_names = ["person", "dog"]
```

Here are a list of metadata keys that's used by builtin features in detectron2.
If you add your own dataset without these metadata, some features may be
unavailable to you.

* `class_names` (list[str]): Used by all instance detection/segmentation tasks.
  A list of names for each instance category. 
  If you load a COCO format dataset, it will be automatically set by the function `load_coco_json`.
  
* `stuff_class_names` (list[str]): Used by semantic/panoptic segmentation.
  A list of names for each stuff category.

* `stuff_colors` (list[tuple(r, g, b)]): Pre-defined color (in [0, 255]) for each stuff category.
  Used by visualization. If not given, will choose randomly.

* `keypoint_names` (list[str]): Used by keypoint localization. A list of names for each keypoint.

* `keypoint_flip_map` (list[tuple[str]]): Used by keypoint localization. A list of pairs of names,
  where each pair are the two keypoints that should be flipped if the image is
  flipped during augmentation.
* `keypoint_connection_rules`: list[tuple(str, str, (r, g, b))]. The colors
  to be used to connect keypoints during visualization.

Here are some more metadata that are specific to the evaluation of certain datasets (e.g. COCO).

* `dataset_id_to_contiguous_id` (dict[int->int]): Used by all instance detection/segmentation tasks in COCO format.
  A mapping from instance class ids in the dataset to a contiguous ids in range [0, #class).
  Will be automatically set by the function `load_coco_json`.

* `stuff_contiguous_id_to_dataset_id` (dict[int->int]): Used when generating prediction json files for
  semantic/panoptic segmentation.
  A mapping from contiguous stuff class ids in [0, #class) to the id in dataset.
  It is useful for evaluation only.

* `json_file`: The COCO annotation json file. Used by COCO evaluation for COCO-format datasets.
* `panoptic_root`, `panoptic_json`: Used by panoptic evaluation.
* `evaluator_type`: Used by the builtin main training script to select
   evaluator. No need to use it if you write your own main script.


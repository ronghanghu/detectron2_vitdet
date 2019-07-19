## The "Detectron2 Dataset Format" for Annotations

For standard tasks
(instance detection, instance/semantic/panoptic segmentation, keypoint detection),
we use a data structure similar to COCO's json annotation
as the basic format to represent a dataset.

The format uses one dict to represent the instance annotations on
one image. The dict may have the following fields.
The fields are often optional, and some functions may be able to
infer certain fields from others if needed, e.g., the data loader
can read from "file_name" if "image" is not available.

+ file_name: the full path to the image file.
+ sem_seg_file_name: the full path to the ground truth semantic segmentation file.
+ image: the image in a numpy array.
+ sem_seg: semantic segmentation ground truth in a 2D numpy array. Values in the array represent
 		semantic labels.
+ height, width: integer. The size of image.
+ image_id (str): a string to identify this image.
		Each dataset may use it for different purposes.
+ annotations (list[dict]): the per-instance annotations of every
		instance in this image. Each annotation dict may contain:
	+ iscrowd: 0 or 1. Whether this instance is labeled as COCO's "crowd region".
	+ bbox (list[float]): list of 4 numbers (x, y, w, h) representing the bounding box of the instance.
	+ bbox_mode (int): the format of bbox.
			It must be a member of [detectron2.structures.BoxMode](detectron2/structures/boxes.py).
		  Currently only supports `BoxMode.XYXY_ABS` and `BoxMode.XYWH_ABS`.
	+ category_id (int): a __positive__ integer in the range [1, num_categories].
	+ segmentation (list[list[float]] or dict):
		+ For `list[list[float]]`, it represents the polygons of
			each object part. Each `list[float]` is one polygon in the
			format of `[x1, y1, ..., xn, yn]`.
			The Xs and Ys are either relative coordinates in [0, 1], or absolute coordinates,
			depend on whether "bbox_mode" is relative.
		+ For `dict`, it represents the per-pixel segmentation in COCO's RLE format.
	+ keypoints (list[float]): in the format of [x1, y1, v1,..., xn, yn, vn].
				v[i] means the visibility of this keypoint.
				`n` must be equal to the number of keypoint categories.
				The Xs and Ys are either relative coordinates in [0, 1], or absolute coordinates,
				depend on whether "bbox_mode" is relative.

				Note that the coordinate annotations in COCO format are integers in range [0, H-1 or W-1].
				They are added by 0.5 to become absolute coordinates.
+ proposal_boxes (array): 2D numpy array with shape (K, 4). K precomputed proposal boxes for this image.
+ proposal_objectness_logits (array): numpy array with shape (K, ), which corresponds to the objectness
        logits of proposals in 'propopsal_boxes'.
+ proposal_bbox_mode (int): the format of the precomputed proposal bbox.
        It must be a member of [detectron2.structures.BoxMode](detectron2/structures/boxes.py).
        Default format is `BoxMode.XYXY_ABS`.


## "Metadata" for Datasets

Each dataset is associated with some metadata, accessible through
`MetadataCatalog.get(dataset_name).some_metadata`.
Metadata is a key-value mapping that contains primitive information that helps interpret what's in the dataset, e.g.,
names of classes, root of files, etc.
This information will be useful for augmentation, evaluation, visualization, logging, etc.
The structure of metadata depends on the what is needed from the corresponding downstream code.

If you register a new dataset through `DatasetRegistry.register`,
you may also want to add its corresponding metadata through
`MetadataCatalog.get(dataset).set(name, value)`, to enable the features that use these metadata.

Here are a list of metadata keys that's used by builtin features in Detectron2.
If you add your own dataset without these metadata, some features may be
unavailable to you.

* `class_names`: Used by all instance detection/segmentation tasks.
  A list of names for each instance category. Will be automatically set by the function `load_coco_json`.

* `dataset_id_to_contiguous_id`: Used by all instance detection/segmentation tasks.
  A mapping from instance class ids in the dataset to a contiguous ids in range [0, #class).
  Will be automatically set by the function `load_coco_json`.

* `stuff_class_names`: Used by semantic/panoptic segmentation.
  A list of names for each stuff category.

* `stuff_contiguous_id_to_dataset_id`: Used by semantic/panoptic segmentation.
  A mapping from contiguous stuff class ids in [0, #class) to the id in dataset.

* `keypoint_names`: Used by keypoint localization. A list of names for each keypoint.

* `keypoint_flip_map`: Used by keypoint localization. A list of pairs of names,
  where each pair are the two keypoints that should be flipped if the image is
  flipped during augmentation.

* `json_file`: Used by COCO evaluation.
* `panoptic_root`, `panoptic_json`: Used by panoptic evaluation.
* `evaluator_type`: Used by the builtin main training script to select
   evaluator. No need to use it if you write your own main script.



## The Detectron2 Dataset format for annotations

For tasks
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

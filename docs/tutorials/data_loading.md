
# The Deafult Data Loading Pipeline

To load a dataset, you can implement your own dataloader and use it in your training loop, as long as it produces something
that your model accepts.
For standard tasks, you may want to use our builtin
`build_detection_{train,test}_loader` which creates a dataloader from config.

Here is how `build_detection_{train,test}_loader` works:

1. It takes the name of the dataset (e.g., "coco_2017_train") from config, and maps it to
	 `list[dict]`.
	 * User can register COCO-format datasets by `detectron2.data.datasets.register_coco_instances()`.
	 * User can register `name->list[dict]` mappings for extra datasets by `DatasetRegistry.register`.
	 * The structure of the dicts is free-form in theory, but should be in "Detectron2 format"
	   if consumed by Detectron2 builtin transformations in the next step.
	 * Details about dataset format and dataset registration in [datasets](datasets).
2. Each dict is mapped by a function ("mapper"):
	 * User can customize this mapping function, by the "mapper" argument in `build_detection_{train,test}_loader`.
	 * The output format is free-form, as long as it is accepted by the consumer of this dataloader
		(usually the model).
		The output format of the default mapper is explained below.
3. The output of the mapper is batched.
	 * Currently it's naively batched to a list.
4. The batched data is the output of the dataloader. And typically it's also the input of
	 `model.forward()`.


### Model Input Format

The output of the default mapper in data loader is a dict.
After batching, it becomes `list[dict]`, one dict per image,
which is the input format of all the builtin models.

The dict may contain the following keys:

* "image": `Tensor` in (C, H, W) format.
* "instances": an `Instances` object, with the following fields:
	+ "gt_boxes": `Boxes` object of N boxes
	+ "gt_classes": `Tensor`, a vector of N labels, in range [0, #class)
	+ "gt_masks": a `PolygonMasks` object, masks for each instance.
	+ "gt_keypoints": a `Keypoints` object, keypoints for each instance.
* "proposals": an `Instance` object used in Fast R-CNN style models, with the following fields:
	+ "proposal_boxes": `Boxes` object of P boxes.
	+ "objectness_logits": `Tensor`, a vector of P scores for each proposal.
* "height", "width": the *desired* output height and width of the image, not necessarily the same
	as the height of width of the input `image`.
	For example, it can be the *original* image height and width before resizing.

	If provided, the model will produce output in this resolution,
	rather than in the resolution of `image`. This is more efficient and accurate.
* "sem_seg": `Tensor[int]` in (H, W) format. The semantic segmentation ground truth.


### Model Output Format

The builtin models produce a `list[dict]`, one dict for each image. Each dict may contain:

* "instances": `Instances` object with the following fields:
	* "pred_boxes": `Boxes` object of N boxes
	* "scores": `Tensor`, a vector of N scores
	* "pred_classes": `Tensor`, a vector of N labels in range [0, #class)
	+ "pred_masks": a `Tensor` of shape (N, H, W), masks for each instance.
	+ "pred_keypoints": a `Tensor` of shape (N, #keypoint, 3).
		Each row in the last dimension is (x, y, score).
* "sem_seg": `Tensor` of (#class, H, W), the semantic segmentation prediction.
* "proposals": TODO
* "panoptic_seg": TODO


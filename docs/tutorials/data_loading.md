
# Use Custom Data Loaders

Detectron2 contains a builtin data loader pipeline.
It's good to understand how it works, if you want to write your own one.

Detectron2 has two functions
`build_detection_{train,test}_loader` which creates a dataloader from a given config.
Here is how `build_detection_{train,test}_loader` work:

1. It takes the name of the dataset (e.g., "coco_2017_train"), and maps it to `list[dict]`.
   Details about dataset format and dataset registration can be found in [datasets](datasets).
2. Each dict is mapped by a function ("mapper"):
	 * User can customize this mapping function, by the "mapper" argument in
        `build_detection_{train,test}_loader`. The default is [DatasetMapper]( ../modules/data.html#detectron2.data.DatasetMapper)
	 * There is no constraints on the output format, as long as it is accepted by the consumer of this dataloader (usually the model).
	The output format of the default mapper is explained below.
3. The outputs of the mapper are batched.
	 * Currently it's naively batched to a list.
4. The batched data is the output of the dataloader. And typically it's also the input of
	 `model.forward()`.


If you want to do something new (e.g., different sampling or batching logic),
You can write your own data loader, as long as it produces the format your model accepts.
Next we explain the input format used by the builtin models in detectron2.


### Model Input Format

The output of the default [DatasetMapper]( ../modules/data.html#detectron2.data.DatasetMapper) in data loader is a dict.
After batching, it becomes `list[dict]`, with one dict per image.
This will be the input format of all the builtin models.

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

The standard models produce a `list[dict]`, one dict for each image. Each dict may contain:

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


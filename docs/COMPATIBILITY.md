

## Compatibility with Detectron

Detectron2 addresses some legacy issues left in Detectron, as a result, their models
are not compatible:
running inference with the same model weights will produce different results.

The major differences are:

- The height and width of a box with corners (x1, y1) and (x2, y2) is now computed as
	width = x2 - x1 and height = y2 - y1;
	In Detectron, a "+ 1" was added both height and width.

	Note that the relevant ops in Caffe2 has [adopted this change of convention](https://github.com/pytorch/pytorch/pull/20550)
	with an extra option.
	So it is still possible to inference a Detectron2-trained model with Caffe2.

	This issue most notably exists in:
	- encoding/decoding in bounding box regression.
	- non-maximum suppression. The effect here is very negligible, though.

- RPN now uses different anchors.

  In Detectron, the anchors were quantized and do not have accurate areas (https://github.com/facebookresearch/Detectron/issues/227).
	In Detectron2, the anchors are center-aligned to feature grid points and not quantized.
	You can enable the old behavior with `ANCHOR_GENERATOR.NAME="OriginalRPNAnchorGenerator"`.
	But we may remove the old behavior in the future.

- Classification layer has different ordering of classes.

	This involves any trainable parameter with shape (..., #categories + 1, ...).
	In Detectron2, [0, K-1] of the total K+1 labels mean the K object categories,
	and label "K" means background.
	In Detectron, lable "0" means background, and [1, K] means the K categories.

- Mask inference function is different.

	In Detectron2, the "paste_mask" function is different and should be more accurate than Detectron.

There are some differences in training as well, but they won't affect model compatibility.
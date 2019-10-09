
## Getting Started with Detectron2

This document provides a brief intro of the usage of builtin tools in detectron2.

For a tutorial that involves actual coding with the API,
see our [Colab Notebook](TODO) which covers how to run inference with an
existing model, and how to train a builtin model on a custom dataset.

For more advanced tutorials, refer to our [documentation](https://detectron2.readthedocs.io).


### Inference with Pre-trained Models

1. Pick a model and its config file from
	[model zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md),
	for example, `mask_rcnn_R_50_FPN_3x.yaml`.
2. Run the demo with
```
python demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --input input1.jpg input2.jpg \
	--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```
It will run the inference and show visualizations in an OpenCV window.

To run on webcam, replace `--input files` with `--webcam`.
To run on a video, replace `--input files` with `--video-input video.mp4`.
To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.


### Train a Standard Model

We provided a script in "tools/train_net.py", that is made to train
all the configs provided in detectron2.
You may want to use it as a reference to write your own training script for a new research.

To train a model with "train_net.py", do:
```
python tools/train_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --num-gpus 8
```

The configs are made for 8-GPU training. To train on 1 GPU, use:
```
python tools/train_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

(Note that we applied the [linear learning rate scaling rule](https://arxiv.org/abs/1706.02677)
when changing the batch size.)

For more options, see `python tools/train_net.py -h`.

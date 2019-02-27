# Detectron in Pytorch (Work in Progress)

This project aims at providing the necessary building blocks for easily
creating detection and segmentation models using PyTorch 1.0.

## Highlights
- **PyTorch 1.0:** RPN, Faster R-CNN and Mask R-CNN implementations that matches or exceeds Detectron accuracies
- **Very fast**: up to **2x** faster than [Detectron](https://github.com/facebookresearch/Detectron) and **30%** faster than [mmdetection](https://github.com/open-mmlab/mmdetection) during training. See [MODEL_ZOO.md](MODEL_ZOO.md) for more details.
- **Memory efficient:** uses roughly 500MB less GPU memory than mmdetection during training
- **Multi-GPU training and inference**
- **Batched inference:** can perform inference using multiple images per batch per GPU
- **CPU support for inference:** runs on CPU in inference time. See our [webcam demo](blob/master/demo) for an example
- Provides pre-trained models for almost all reference Mask R-CNN and Faster R-CNN configurations with 1x schedule.

## Simple Demo

We provide a simple webcam demo that illustrates how you can use `detectron2` for inference:
```bash
cd demo
# use input files:
python demo.py --config-file ../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.py \
  --input /path/to/images*.jpg --output /path/to/output/directory
# use webcam:
python demo.py --config-file ../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.py --webcam
```

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.


## Model Zoo and Baselines

Pre-trained models, baselines and comparison with Detectron and mmdetection
can be found in [MODEL_ZOO.md](MODEL_ZOO.md)

Models will be downloaded to `$TORCH_MODEL_ZOO`, which defaults to `~/.torch/models`.

## Perform training on COCO dataset

For the following examples to work, you need to first install `detectron2`.

You will also need to download the COCO dataset.
We recommend to symlink the path to the coco dataset to `datasets/` as follows

We use `minival` and `valminusminival` sets from [Detectron](https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/data/README.md#coco-minival-annotations)

```bash
# symlink the coco dataset
cd ~/github/maskrcnn-benchmark
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2014 datasets/coco/train2014
ln -s /path_to_coco_dataset/test2014 datasets/coco/test2014
ln -s /path_to_coco_dataset/val2014 datasets/coco/val2014
```

You can also configure your own paths to the datasets.
For that, all you need to do is to modify `detectron2/config/paths_catalog.py` to
point to the location where your dataset is stored.
You can also create a new `paths_catalog.py` file which implements the same two classes,
and pass it as a config argument `PATHS_CATALOG` during training.

### Single GPU training

```bash
python /path_to_detectron2/tools/train_net.py --config-file "/path/to/config/file.yaml"
```

## Troubleshooting
If you have issues running or compiling this code, we have compiled a list of common issues in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). If your issue is not present there, please feel
free to open a new issue.

## License

maskrcnn-benchmark is released under the MIT license. See [LICENSE](LICENSE) for additional details.

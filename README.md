# Detectron in Pytorch (Work in Progress)

This project aims at providing the necessary building blocks for easily
creating detection and segmentation models using PyTorch 1.0.

## Highlights
- **PyTorch 1.0:** RPN, Faster R-CNN and Mask R-CNN implementations that matches or exceeds Detectron accuracies
- **Very fast**: up to **2x** faster than [Detectron](https://github.com/facebookresearch/Detectron) and **30%** faster than [mmdetection](https://github.com/open-mmlab/mmdetection) during training. See [MODEL_ZOO.md](MODEL_ZOO.md) for more details.
- **Memory efficient:** uses roughly 500MB less GPU memory than mmdetection during training
- **Multi-GPU training and inference**
- **Batched inference:** can perform inference using multiple images per batch per GPU
- **CPU support for inference:** runs on CPU in inference time. See our [webcam demo](demo) for an example
- Provides pre-trained models for almost all reference Mask R-CNN and Faster R-CNN configurations with 1x schedule.

## Simple Demo

We provide a simple demo that illustrates how you can use `detectron2` for inference:
```bash
cd demo
# use input files:
python demo.py --config-file ../configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml \
  --input /path/to/images*.jpg --output /path/to/output/directory
# use webcam:
python demo.py --config-file ../configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml --webcam
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
See methods in "DatasetCatalog".

### Single GPU training

```bash
python /path_to_detectron2/tools/train_net.py --config-file "/path/to/config/file.yaml"
```

## Environment variables

- `DETECTRON2_ENV_MODULE`: Name of a module defining a function called `setup_environment` to call before running any detectron2 code. This function can be used to perform set up steps that are specific to different computing or cluster environments.

## Troubleshooting
If you have issues running or compiling this code, we have compiled a list of common issues in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). If your issue is not present there, please feel
free to open a new issue.

## Major Compatibility Differences Compared to Detectron (v1)

- The height and width of a box with corners (x1, y1) and (x2, y2) is computed as width = x2 - x1 and height = y2 - y1; in Detectron v1 a "+ 1" was added both height and width. This makes detectron2 incompatible with models trained using Detectron v1.
- For a dataset with K object categories, these categories are assigned labels [0, K - 1]. If a background category is used (e.g., with a classifier that uses softmax), the background category is assigned label K. In Detectron v1 background was assigned label 0 and the object classes labels [1, K].
- RPN anchors are now center-aligned to feature grid points and they are not quantized. If one loads a model trained using the original RPN anchor definition, then `MODEL.RPN.ANCHOR_GENERATOR_NAME` must be set to `OriginalRPNAnchorGenerator`.

## License

maskrcnn-benchmark is released under the MIT license. See [LICENSE](LICENSE) for additional details.

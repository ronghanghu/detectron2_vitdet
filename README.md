# Detectron2

Detectron2 is Facebook AI Research's next generation software system
that implements state-of-the-art object detection algorithms.
It is a ground-up rewrite of the previous version,
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).
It is written in Python and powered by the [PyTorch](https://pytorch.org) deep
learning framework.

![logo](.github/Detectron2-Logo-Horz.svg)


## Installation


## Quick Start



## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).


## License

Detectron2 is released under the [Apache 2.0 license](https://github.com/facebookresearch/detectron2/blob/master/LICENSE).

## Citing Detectron

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```


=====================================

old doc:


## Simple Demo

We provide a simple demo that illustrates how you can use `detectron2` for inference:
```bash
cd demo
# use input files:
python demo.py --config-file ../configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml \
  --input /path/to/images*.jpg --output /path/to/output/directory
# use webcam:
python demo.py --config-file ../configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml --webcam
```

### Single GPU training

```bash
python /path_to_detectron2/tools/train_net.py --config-file "/path/to/config/file.yaml"
```

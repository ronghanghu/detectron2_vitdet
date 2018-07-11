# Object Detection and Segmentation in PyTorch

This project aims at providing the necessary building blocks for easily
creating detection and segmentation models.

## Installation

### Requirements:
- PyTorch compiled from master. Installation instructions can be found in [here](https://github.com/pytorch/pytorch#installation).
- torchvision from master
- cocoapi

### Installing the lib

```bash
git clone git@github.com:fairinternal/detectron.pytorch.git
cd detectron.pytorch

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop
```

### Step by step installation on the FAIR Cluster

```bash
# setup environments
module load anaconda3/5.0.1 cuda/9.0 cudnn/v7.0-cuda.9.0 NCCL/2.2.13-cuda.9.0

conda create --name pytorch_detection
source activate pytorch_detection

conda install ipython

# pytorch and coco api dependencies
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing cython
conda install -c mingfeima mkldnn
conda install -c pytorch magma-cuda90

pip install ninja

# cloning and installing PyTorch from master
mkdir ~/github && cd ~/github
git clone --recursive git@github.com:pytorch/pytorch.git
cd pytorch
# compile for several GPU architectures
TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;6.1;7.0" python setup.py build develop

# install torchvision
cd ~/github
git clone git@github.com:pytorch/vision.git
cd vision
python setup.py install

# install pycocotools
cd ~/github
git clone git@github.com:cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd ~/github
git clone git@github.com:fairinternal/detectron.pytorch.git
cd detectron.pytorch
TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;6.1;7.0" python setup.py build develop

# symlink the coco dataset
cd ~/github/detectron.pytorch
ln -s /datasets01/COCO/060817/annotations configs/datasets/coco/annotations
ln -s /datasets01/COCO/060817/train2014 configs/datasets/coco/train2014
ln -s /datasets01/COCO/060817/test2014 configs/datasets/coco/test2014
ln -s /datasets01/COCO/060817/val2014 configs/datasets/coco/val2014
ln -s /private/home/fmassa/imagenet_detectron_models configs/models
# TODO: coco test 2015 and coco test 2017
```

## Running training code

For the following examples to work, you need to first install `torch_detectron`.

### Single GPU training

```bash
python -m torch_detectron.train --config-file "/path/to/config/file.py"
```

### Multi-GPU training
We use internally `torch.distributed.launch` in order to launch
multi-gpu training. This utility function from PyTorch spawns as many
Python processes as the number of GPUs we want to use, and each Python
process will only use a single GPU.

```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_detectron/train.py --config-file "path/to/config/file.py"
```

It is unfortunately not possible (at least I haven't figured out)
to launch two processes in python with `-m` flag, so that in the
multi-gpu case we need to specify the full path of the detectron `train.py`
file.

## Model Zoo and Baselines

The numbers below were obtaining by running the code on 8 V100 GPUs, using Cuda 8.0 and CUDNN 6.0.

### RPN Proposal Baselines

backbone | total train time (h) | train time (s / it) | test time (s / it) | max memory (GB) | mAR | model id
-- | -- | -- | -- | -- | -- | --
R-50-C4  | 6.2 | 0.246 | 0.054 | 3.2 | 51.6 | 4457116
R-50-FPN | 9.9 | 0.397 | 0.067 | 5.2 | 56.8 | 4409548

*NOTE* For R-50-C4, my original run had wrong test hyperparameters (`PRE_NMS_TOP_N` and `POST_NMS_TOP_N` where smaller than they should be),
so the corrected mAR numbers were obtaining on a separate test run.

### End to end detection and mask

backbone | type | lr sched | im / gpu | train mem(GB) | train time (s/iter) | total train time(hr) | inference time(s/im) | box AP | mask AP
-- | -- | -- | -- | -- | -- | -- | -- | -- | --
R-50-C4  | Fast | 1x | 1 |  4.0 | 0.409  | 20.4 | 0.140         | 34.3 | -    | 4022836
R-50-FPN | Fast | 1x | 2 |  5.3 | 0.464  | 11.5 | 0.102         | 36.3 | -    | 4418514
R-50-C4  | Mask | 1x | 1 |  4.2 | 0.486  | 24.3 | 0.151 + 0.022 | 35.5 | 31.4 | 4354304
R-50-FPN | Mask | 1x | 2 |  7.5 | 0.565  | 14.1 | 0.110 + 0.026 | 37.2 | 34.0 | 4444352

#### Comparing with Detectron C2 using CUDA 9.0 and CUDNN 7.0

In the following table, we run a few detection algorithms for 20 minutes using CUDA 9.0 and CUDNN 7.0 on
8 V100 cards, using both our implementation and the Detectron Caffe2 codebase.

backbone | type | im / gpu | train mem(GB) | train time (s/iter) | Codebase
-- | -- | -- | -- | -- | --
R-50-FPN | Fast | 2 | 5.2 | 0.403 | This repo
R-50-FPN | Fast | 2 | 7.4 | 0.488 | Detectron C2
R-50-FPN | Mask | 2 | 7.2 | 0.459 | This repo
R-50-FPN | Mask | 2 | 8.7 | 0.796 | Detectron C2

As can be seem from this table, the implementation in this repo uses significantly less memory
and is significantly faster. Also, we use a naive Python implementation for `AffineChannel`; an optimized
implementation would bring additional speed benefits.

## Abstractions
The main abstractions introduced by `torch_detectron` that are useful to
have in mind are the following:

### ImageList
In PyTorch, the first dimension of the input to the network generally represents
the batch dimension, and thus all elements of the same batch have the same
height / width.
In order to support images with different sizes and aspect ratios in the same
batch, we created the `ImageList` class, which holds internally a batch of
images (os possibly different sizes). The images are padded with zeros such that
they have the same final size and batched over the first dimension. The original
sizes of the images before padding are stored in the `image_sizes` attribute,
and the batched tensor in `tensors`.
We provide a convenience function `to_image_list` that accepts a few different
input types, including a list of tensors, and returns an `ImageList` object.

```python
from torch_detectron.core.image_list import to_image_list

images = [torch.rand(3, 100, 200), torch.rand(3, 150, 170)]
batched_images = to_image_list(images)

# it is also possible to make the final batched image be a multiple of a number
batched_images_32 = to_image_list(images, size_divisible=32)
```

### BBox
The `BBox` class holds a set of bounding boxes (represented as a `Nx4` tensor) for
a specific image, as well as the size of the image as a `(width, height)` tuple.
It also contains a set of methods that allow to perform geometric
transformations to the bounding boxes (such as cropping, scaling and flipping).
The class accepts bounding boxes from two different input formats:
- `xyxy`, where each box is encoded as a `x1`, `y1`, `x2` and `y2` coordinates)
- `xywh`, where each box is encoded as `x1`, `y1`, `w` and `h`.

Additionally, each `BBox` instance can also hold arbitrary additional information
for each bounding box, such as labels, visibility, probability scores etc.

Here is an example on how to create a `BBox` from a list of coordinates:
```python
from torchvision.structures.bounding_box import BBox, FLIP_LEFT_RIGHT

width = 100
height = 200
boxes = [
  [0, 10, 50, 50],
  [50, 20, 90, 60],
  [10, 10, 50, 50]
]
# create a BBox with 3 boxes
bbox = BBox(boxes, size=(width, height), mode='xyxy')

# perform some box transformations, has similar API as PIL.Image
bbox_scaled = bbox.resize((width * 2, height * 3))
bbox_flipped = bbox.transpose(FLIP_LEFT_RIGHT)

# add labels for each bbox
labels = torch.tensor([0, 10, 1])
bbox.add_field('labels', labels)

# bbox also support a few operations, like indexing
# here, selects boxes 0 and 2
bbox_subset = bbox[[0, 2]]
```

### Anchor Generator

### Box Selectors

### Box Coder

### Post Processors

### Training

#### Matcher

#### Positive Negative Sampler

#### Target Preparator

#### Loss Computation

- Region Proposals
- Anchor Generators
- Box Selectors
- Post Processors
- FPN / Mask utilities

*TODO: explain the different abstractions and add a figure*


## The configuration file
Detection models are very modular and have many degrees of freedom.
In order to manage easy experimentation, we provide a flexible configuration
system that has a similar feel as the one used in Detectron C2, but that
allows for more possibilities for configuration as it is a python file,
and not a yaml.

The configuration object can be seen as a tree, where each node is an object
(potentially callable, or a primitive type) that can be modified and the
modifications will be propagated to the root of the tree (the config object).
This enables both modifying a single argument (learning rate) without having
to specify the full optimizer, but also gives the flexibility to completely
replace the default optimizer -- the user just needs to provide a callable
that returns the optimizer.

The user can specify an arbitrary callable to each node of the tree, and
each callable can have arguments as well (which can by themselves be
callables), so that one can modify any node and have the information
propagate up in the config.

To give an experience closer to manipulating the yaml from Detectron C2,
the config object can be nicely printed (for logging purposes) and
manipulated. An example of the print is shown below:

```python
...
  'TRAIN': <torch_detectron.helpers.func_repr.AttrDict object at 0x7f5ef89b2a40>
    'DATA': <torch_detectron.helpers.data._Data object at 0x7f5ef8993cc0>:
      'DATASET': <torch_detectron.helpers.data._COCODataset object at 0x7f5ef8993cf8>:
        'FILES': [('/datasets01/COCO/060817/annotations/instances_train2014.json', '/datasets01/COCO/060817/train2014/')]
      'DATALOADER': <torch_detectron.helpers.data._DataLoader object at 0x7f5ef89b8710>:
        'NUM_WORKERS': 4
        'IMAGES_PER_BATCH': 1
        'SAMPLER': <torch_detectron.helpers.data._DataSampler object at 0x7f5ef89b8ef0>:
          'SHUFFLE': True
          'DISTRIBUTED': False
        'COLLATOR': <torch_detectron.helpers.data._Collator object at 0x7f5f41f46080>:
          'SIZE_DIVISIBLE': 0
      'TRANSFORM': <torch_detectron.helpers.data._DataTransform object at 0x7f5f41f462b0>:
        'MIN_SIZE': 800
        'MAX_SIZE': 1333
        'MEAN': [102.9801, 115.9465, 122.7717]
        'STD': [1, 1, 1]
        'TO_BGR255': True
        'FLIP_PROB': 0.5
...
```

More importantly, this config system tries to decouple the way the
arguments are passed and the implementation itself, the goal being
that we can import this package as a library, and end users don't
need to ship the whole repo in order to share their code.

### Examples:

#### Simple case: modifying a few hyperparameters

The default configuration file in `helpers/config.py` already contains
reasonable set of defaults for training Faster R-CNN models.

If the user wants to modify only a few hyper-parameters of the training,
it can be accomplished in a very similar way as the one done by
Detectron C2.

Here is an example where we specify a different learning rate,
the path to our local COCO dataset and a few hyperparameters of
our model:
```python
from torch_detectron.helpers.config import get_default_config


config = get_default_config()

# we specify a list of tuples, each tuple containing
# the information for a single COCO dataset.
# The first element of the tuple is the json file with
# the annotations, and the second the folder containing
# the images.
# note that we can specify arbitrary subsets of the dataset
# which will be concatenated together.
config.TRAIN.DATA.DATASET.FILES = [
        ('/datasets01/COCO/060817/annotations/instances_train2014.json',
         '/datasets01/COCO/060817/train2014/'),
        ('/private/home/fmassa/coco_trainval2017/annotations/instances_valminusminival2017.json',
         '/datasets01/COCO/060817/val2014/')
]

# and the testing dataset
config.TEST.DATA.DATASET.FILES = [
        ('/private/home/fmassa/coco_trainval2017/annotations/instances_val2017_mod.json',
         '/datasets01/COCO/060817/val2014/')
]

# now change a few things about the model
config.MODEL.RPN.PRE_NMS_TOP_N = 6000
config.MODEL.RPN.POST_NMS_TOP_N = 1000

# now modify the optimizer
config.SOLVER.MAX_ITER = 180000
lr = 0.01

config.SOLVER.OPTIM.BASE_LR = lr
# make a different learning rate for bias
# following C2 behaviour
config.SOLVER.OPTIM.BASE_LR_BIAS = 2 * lr  # can use arbitrary python code
```

#### Advanced examples: using `torch_detectron` as a library

If you want to build a model that does not exactly fit the constraints
that are imposed by the model builder that we provide, you can still
have the flexibility to modify arbitrary levels of the configuration
file, while having the new config being propagated to the top level.

1. Adding your own dataset

Suppose that you want to use a new dataset which
does not fit with the COCO style. You can do it in the following manner:

```python
from torch_detectron.helpers.config import get_default_config
# this is just a helper class, it's not strictly needed
from torch_detectron.helpers.config_utils import ConfigClass

class MyDataset(object):
    def __init__(self, transforms):
        ...
    def __getitem__(self, idx):
        ...
        # image is a torch.Tensor
        # and target is a BBox object
        return image, target
    def __len__(self):
        return ...

class MyDatasetBuilder(ConfigClass):
    def __call__(self, transforms):
        return MyDataset(transforms)

config = get_default_config()
config.TRAIN.DATA.DATASET = MyDatasetBuilder()
```
There are no other constraints imposed. The constraints on the image
and the target are only necessary if you want to use the default
`collate_fn` that we provide, but you can provide your own as well.

2. Modifying the optimizer

The default optimizer that we provide uses SGD. Creating a new
optimizer is simple:

```python
from torch_detectron.helpers.config import get_default_config
from torch_detectron.helpers.config_utils import ConfigClass

from torch.optim import Adam

class MyOptimizer(ConfigClass):
    def __call__(self, model):
        lr = self.LEARNING_RATE
        optimizer = Adam(model.parameters(), lr=lr)
        return optimizer

config = get_default_config()
config.SOLVER.OPTIM = MyOptimizer()
config.SOLVER.OPTIM.LEARNING_RATE = 0.01
```

3. Creating a whole new model

Following the approach that we have just presented, it is possible
to modify individual blocks of the MODEL (for example the `HEADS` or
the `RPN`) in order to implement your model.
But if you want to create a completely new architecture which has
never been tried before, you can do it without having to modify the
`torch_detectron` files (but you _can_, if you want).
Here is a toy example:
```python
from torch_detectron.helpers.config import get_default_config
from torch_detectron.helpers.config_utils import ConfigClass

from torch_detectron.core.image_list import to_image_list

from torch_detectron.model_builder.resnet import resnet50_conv4_body
from torch_detectron.core.anchor_generator import AnchorGenerator
from torch_detectron.core.rpn_losses import (RPNLossComputation,
            RPNTargetPreparator)
from torch_detectron.core.proposal_matcher import Matcher
from torch_detectron.core.box_selector import RPNBoxSelector
from torch_detectron.core.balanced_positive_negative_sampler import (
            BalancedPositiveNegativeSampler)
from torch_detectron.core.box_coder import BoxCoder

from torch import nn

class MyModel(nn.Module):
    def __init__(self, pretrained_weights):
        super(MyModel, self).__init__()
        self.backbone = resnet50_conv4_body(pretrained=pretrained_weights)
        self.refine_module = nn.Conv2d(1024, 1024, 3, 1, 1)

        self.anchor_generator = AnchorGenerator(
            scales=(0.125, 0.25, 0.5, 1., 2.))

        self.cls_scores = nn.Conv2d(1024, 5 * 3, 1)
        self.box_reg = nn.Conv2d(1024, 5 * 3 * 4, 1)

        # Those are classes that helps matching / sampling and creating
        # the targets for the detection models
        matcher = Matcher(0.7, 0.3, force_match_for_each_row=True)
        box_coder = BoxCoder(weights=(1., 1., 1., 1.))
        target_preparator = RPNTargetPreparator(matcher, box_coder)
        fg_bg_sampler = BalancedPositiveNegativeSampler(
                batch_size_per_image=256, positive_fraction=0.5)
        # this class is in charge of computing the losses for the RPN
        self.rpn_loss_evaluator = RPNLossComputation(target_preparator, fg_bg_sampler)

        # for inference only
        self.box_selector = RPNBoxSelector(12000, 2000, 0.7, 0)

    def forward(self, images, target=None):
        # convert the input images to an ImageList, if not already
        images = to_image_list(images)
        # get features
        features = self.backbone(images.tensors)
        # compute anchors from image and features
        anchors = self.anchor_generator(images.image_sizes, features)

        # some arbitrary new criteria is added
        if features.mean() > 0.5:
            features = self.refine_module(features)

        # compute objectness and regression
        objectness = self.cls_scores(features)
        rpn_box_regression = self.box_reg(features)

        # behavior of the model changes during training and testing
        # during training, we return the losses, during testing the predicted boxes
        if not self.training:
            result = self.box_selector(anchors, objectness, rpn_box_regression)
            return result[0]  # returns result for single feature map


        # loss computation is handled in loss_evaluator
        loss_objectness, loss_box_reg = self.rpn_loss_evaluator(
                anchors, objectness, rpn_box_regression, targets)

        return dict(
                    loss_objectness=loss_objectness,
                    loss_box_reg=loss_box_reg)

# can add arbitrary attributes so that it can be configurable
class ModelGetter(ConfigClass):
    def __call__(self):
        return MyModel(self.WEIGHTS)

config = get_default_config()
config.MODEL = ModelGetter()
config.MODEL.WEIGHTS = '/path/to/pretrained.pkl'
```
The current constraint we impose to the model is that in `.train()` mode
the mode should return a dict containing the losses, while in `.eval()` mode
the model returns a `BBox` object with extra attributes (like `scores` or
`masks`).

## Style guide

```bash
cd ~/github/detectron.pytorch

# First run isort
isort -rc -sl -t 1 --atomic -p torch_detectron -o torch -o torchvision .

# Then run black
black --exclude configs/datasets .
```

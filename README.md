# Object Detection and Segmentation in PyTorch

This project aims at providing the necessary building blocks for easily
creating detection and segmentation models.

## Abstractions

- Region Proposals
- Anchor Generators
- Box Selectors
- Post Processors
- FPN / Mask utilities

*TODO: explain the different abstractions and add a figure*

## Installation

### Requirements:
- PyTorch compiled from master
- torchvision from master
- cocoapi

### Installing the lib

```
git clone git@github.com:fairinternal/detectron.pytorch.git
cd detectron.pytorch

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you wantand won't need to
# re-build it
python setup.py build develop
```

## Running training code

For the following examples to work, you need to first install `torch_detectron`.

### Single GPU training

```
python -m torch_detectron.train --config-file "/path/to/config/file.py"
```

### Multi-GPU training
We use internally `torch.distributed.launch` in order to launch
multi-gpu training. This utility function from PyTorch spawns as many
Python processes as the number of GPUs we want to use, and each Python
process will only use a single GPU.

```
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_detectron/train.py --config-file "path/to/config/file.py"
```

It is unfortunately not possible (at least I haven't figured out)
to launch two processes in python with `-m` flag, so that in the
multi-gpu case we need to specify the full path of the detectron `train.py`
file.

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

```
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
from torch_detectron.helpers.config import config

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
config.MODEL.REGION_PROPOSAL.PRE_NMS_TOP_N = 6000
config.MODEL.REGION_PROPOSAL.POST_NMS_TOP_N = 1000

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
from torch_detectron.helpers.config import config
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

config.TRAIN.DATA.DATASET = MyDatasetBuilder()
```
There are no other constraints imposed. The constraints on the image
and the target are only necessary if you want to use the default
`collate_fn` that we provide, but you can provide your own as well.

2. Modifying the optimizer

The default optimizer that we provide uses SGD. Creating a new
optimizer is simple:

```python
from torch_detectron.helpers.config import config
from torch_detectron.helpers.config_utils import ConfigClass

from torch.optim import Adam

class MyOptimizer(ConfigClass):
    def __call__(self, model):
        lr = self.LEARNING_RATE
        optimizer = Adam(model.parameters(), lr=lr)
        return optimizer

config.SOLVER.OPTIM = MyOptimizer()
config.SOLVER.OPTIM.LEARNING_RATE = 0.01
```

3. Creating a whole new model

Following the approach that we have just presented, it is possible
to modify individual blocks of the MODEL (for example the `HEADS` or
the `REGION_PROPOSAL`) in order to implement your model.
But if you want to create a completely new architecture which has
never been tried before, you can do it without having to modify the
`torch_detectron` files (but you _can_, if you want).
Here is a toy example:
```python
from torch_detectron.helpers.config import config
from torch_detectron.helpers.config_utils import ConfigClass

from torch_detectron.core.image_list import to_image_list
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
    def __init__(self):
        super(MyModel, self).__init__()
        self.anchor_generator = AnchorGenerator(
            scales=(0.125, 0.25, 0.5, 1., 2.))
        self.backbone = ...
        self.my_crazy_module = ...

        self.cls_scores = nn.Conv2d(...)
        self.box_reg = nn.Conv2d(...)
        ...

        # Those are classes that helps matching / sampling and creating
        # the targets for the detection models
        matcher = Matcher(0.7, 0.3, force_match_for_each_row=True)
        box_coder = BoxCoder(weights=(1., 1., 1., 1.))
        target_preparator = RPNTargetPreparator(matcher, box_coder)
        fg_bg_sampler = BalancedPositiveNegativeSampler(
                batch_size_per_image=256, positive_fraction=0.5)
        self.rpn_loss_evaluator = RPNLossComputation(target_preparator, fg_bg_sampler)

        # for inference only
        self.box_selector = RPNBoxSelector(12000, 2000, 0.7, 0)

    def forward(self, images, target=None):
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        anchors = self.anchor_generator(images.image_sizes, features)

        if features.mean() > 0.5:
            features = self.my_crazy_module(features)

        objectness = self.cls_scores(features)
        rpn_box_regression = self.box_reg(features)

        if not self.training:
            result = self.box_selector(anchors, objectness, rpn_box_regression)
            return result[0]  # returns result for single feature map


        loss_objectness, loss_box_reg = self.rpn_loss_evaluator(
                anchors, objectness, rpn_box_regression, targets)

        return dict(
                    loss_objectness=loss_objectness,
                    loss_box_reg=loss_box_reg)
        
```
The current constraint we impose to the model is that in `.train()` mode
the mode should return a dict containing the losses, while in `.eval()` mode
the model returns a `BBox` object with extra attributes (like `scores` or
`masks`).

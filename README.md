# Object Detection and Segmentation in PyTorch

This project aims at providing the necessary building blocks for easily
creating detection and segmentation models.

### Note: For the previous implementation of this code, see https://github.com/facebookexternal/detectron.pytorch/tree/my_proposal_branch

## Installation

### Requirements:
- PyTorch compiled from master. Installation instructions can be found in [here](https://github.com/pytorch/pytorch#installation).
- torchvision from master
- cocoapi

### Installing the lib

```bash
git clone git@github.com:facebookexternal/detectron.pytorch.git
cd detectron.pytorch

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop
```

### Step-by-step installation on the FAIR Cluster

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
python /path_to_detectron/tools/train_net.py --config-file "/path/to/config/file.yaml"
```

### Multi-GPU training
We use internally `torch.distributed.launch` in order to launch
multi-gpu training. This utility function from PyTorch spawns as many
Python processes as the number of GPUs we want to use, and each Python
process will only use a single GPU.

```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_detectron/tools/train_net.py --config-file "path/to/config/file.yaml"
```

It is unfortunately not possible (at least I haven't figured out)
to launch two processes in python with `-m` flag, so that in the
multi-gpu case we need to specify the full path of the detectron `train.py`
file.

## FB infra usage

This section includes instructions that are specific to using the code on FB infra (devgpus and cluster).

### Checking out on a devgpu

This repo is not yet integrated into fbcode. In the meantime, we adopt a workaround in
which one creates an fbcode checkout and then "grafts" the git repo on top of it. Buck
`TARGETS` files are implemented such that once the git repo is checked out in the correct
location (`//experimental/deeplearning/vision/detectron_pytorch`) it will build correctly
using all necessarily dependencies from fbcode, tp2, pyfi, etc.

```bash
# Create a separate fbcode checkout that we'll use for "grafting" the github repo on top of
fbclone fbsource fbsource-github-grafts

# Move to the grafting location
cd /data/users/$USER/fbsource-github-grafts/fbcode/experimental/deeplearning/vision/

# Clone the git repo
git clone git@github.com:facebookexternal/detectron.pytorch.git
# The buck targets files require a slightly different name
mv detectron.pytorch detectron_pytorch
```

### Running on a devgpu

**NOTE: multi-gpu support is currently broken due to an issue in c10d that is under investigation**

To build:

```bash
buck build @mode/dev-nosan -c python.native_link_strategy=separate //experimental/deeplearning/vision/detectron_pytorch/...
```

To run:

```bash
cd /data/users/$USER/fbsource-github-grafts/fbcode/experimental/deeplearning/vision/detectron_pytorch

TORCH_DETECTRON_ENV_MODULE=infra/fb/env.py \
  /data/users/$USER/fbsource-github-grafts/fbcode/buck-out/gen/experimental/deeplearning/vision/detectron_pytorch/tools/train_net.par \
  --config-file configs/e2e_mask_rcnn_R_50_C4_1x.yaml \
  PATHS_CATALOG infra/fb/paths_catalog.py
```

Note the two FB infra specific settings:
 - Setting `TORCH_DETECTRON_ENV_MODULE` such that an FB infra specific environment setup function can execute
 - Setting `PATHS_CATALOG` so that datasets and models can be found in FB specific locations

### Running on the FB GPU cluster

**NOTE: multi-gpu support is currently broken due to an issue in c10d that is under investigation**

```bash
GPU=1 MEM=24 CPU=8 ./infra/fb/launch.sh configs/quick_schedules/e2e_faster_rcnn_R_50_FPN_quick.yaml test
```

## Model Zoo and Baselines

Content coming soon. For now, refer to https://github.com/facebookexternal/detectron.pytorch/pull/60


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
from torch_detectron.structures.image_list import to_image_list

images = [torch.rand(3, 100, 200), torch.rand(3, 150, 170)]
batched_images = to_image_list(images)

# it is also possible to make the final batched image be a multiple of a number
batched_images_32 = to_image_list(images, size_divisible=32)
```

### BoxList
The `BoxList` class holds a set of bounding boxes (represented as a `Nx4` tensor) for
a specific image, as well as the size of the image as a `(width, height)` tuple.
It also contains a set of methods that allow to perform geometric
transformations to the bounding boxes (such as cropping, scaling and flipping).
The class accepts bounding boxes from two different input formats:
- `xyxy`, where each box is encoded as a `x1`, `y1`, `x2` and `y2` coordinates)
- `xywh`, where each box is encoded as `x1`, `y1`, `w` and `h`.

Additionally, each `BoxList` instance can also hold arbitrary additional information
for each bounding box, such as labels, visibility, probability scores etc.

Here is an example on how to create a `BoxList` from a list of coordinates:
```python
from torch_detectron.structures.bounding_box import BoxList, FLIP_LEFT_RIGHT

width = 100
height = 200
boxes = [
  [0, 10, 50, 50],
  [50, 20, 90, 60],
  [10, 10, 50, 50]
]
# create a BoxList with 3 boxes
bbox = BoxList(boxes, size=(width, height), mode='xyxy')

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

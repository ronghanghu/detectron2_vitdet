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

### Step-by-step installation on a devgpu

```bash
# Initial steps from https://our.internmc.facebook.com/intern/wiki/Caffe2Guide/open-source/
alias with-proxy="HTTPS_PROXY=http://fwdproxy.any:8080 HTTP_PROXY=http://fwdproxy.any:8080 FTP_PROXY=http://fwdproxy.any:8080 https_proxy=http://fwdproxy.any:8080 http_proxy=http://fwdproxy.any:8080 ftp_proxy=http://fwdproxy.any:8080 http_no_proxy='\''*.facebook.com|*.tfbnw.net|*.fb.com'\'"

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

sudo cp -R /home/engshare/third-party2/cudnn/7.1.2/src/cuda/include/* /usr/local/cuda/include/
sudo cp -R /home/engshare/third-party2/cudnn/7.1.2/src/cuda/lib64/* /usr/local/cuda/lib64/

mkdir ~/local/github

# Obtain NCCL2 from https://developer.nvidia.com/nccl
# E.g., download nccl_2.2.13-1+cuda9.2_x86_64.txz
mkdir ~/local/github/nccl2
cd ~/local/github/nccl2
mv nccl_2.2.13-1+cuda9.2_x86_64.txz .
tar --no-same-owner -xvf nccl_2.2.13-1+cuda9.2_x86_64.txz
sudo mv nccl_2.2.13-1+cuda9.2_x86_64/include/* /usr/local/cuda/include/
sudo cp -P nccl_2.2.13-1+cuda9.2_x86_64/lib/libnccl* /usr/local/cuda/lib64/
ldconfig

# Install anaconda
cd ~/local

with-proxy wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O anaconda3.sh
chmod +x anaconda3.sh
with-proxy ./anaconda3.sh -b -p ~/local/anaconda3

# Activate anaconda
. ~/local/anaconda3/bin/activate

# Create a virtual environment to work in
with-proxy conda create --prefix ~/local/conda/pytorch_detectron
conda activate /home/rbg/local/conda/pytorch_detectron

conda install ipython

# pytorch and coco api dependencies
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing cython
conda install -c mingfeima mkldnn
conda install -c pytorch magma-cuda90

pip install ninja

# cloning and installing PyTorch from master
mkdir ~/local/github && cd ~/local/github
git clone --recursive git@github.com:pytorch/pytorch.git
cd pytorch

# compile for several GPU architectures using NCCL2
NCCL_ROOT_DIR=/usr/local/cuda/ TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;6.1;7.0" \
  python setup.py build develop

# install torchvision
cd ~/local/github
git clone git@github.com:pytorch/vision.git
cd vision
python setup.py install

# install pycocotools
cd ~/local/github
git clone git@github.com:cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd ~/local/github
git clone git@github.com:fairinternal/detectron.pytorch.git
cd detectron.pytorch
TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;6.1;7.0" python setup.py build develop

# links to datasets and models
cd ~/local/github/detection.pytorch
ln -s /data/local/packages/ai-group.coco_train2014/latest/coco_train2014 configs/datasets/coco/train2014
ln -s /data/local/packages/ai-group.coco_val2014/latest/coco_val2014 configs/datasets/coco/val2014
ln -s /mnt/vol/gfsai-east/ai-group/datasets/json_dataset_annotations/coco configs/datasets/coco/annotations
mkdir configs/models
ln -s /mnt/vol/gfsai-east/ai-group/users/vision/torch_detectron_models/r50_new.pth configs/models/R-50.pth
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

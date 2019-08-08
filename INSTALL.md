## Installation

### Requirements:
- Python 3
- PyTorch 1.0 from a nightly release. Installation instructions can be found in https://pytorch.org/get-started/locally/
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- (optional) OpenCV for the webcam demo


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name detectron2
source activate detectron2

# this installs the right pip and dependencies for the fresh python
conda install ipython

# detectron2 and coco api dependencies
pip install ninja yacs cython matplotlib

# Follow PyTorch installation in https://pytorch.org/get-started/locally/
# For example:
conda install pytorch-nightly -c pytorch

# install pycocotools
pip install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# optionally, install cityscapescripts
pip install --user cityscapescripts

# optionally, install shapely to use cropping for instance segmentation or work with cityscapes'
# ground truth JSON files
pip install --user shapely

# Clone this repo, and run:
python setup.py build develop
# This will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```

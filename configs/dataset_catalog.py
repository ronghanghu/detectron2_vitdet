# Modified from facebookresearch/Detectron
#
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Collection of available datasets."""

import os

# Path to data dir
_DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")

# Required dataset entry keys
_IM_DIR = "image_directory"
_ANN_FN = "annotation_file"

# Available datasets
_DATASETS = {
    "coco_2014_train": {
        _IM_DIR: _DATA_DIR + "/coco/train2014",
        _ANN_FN: _DATA_DIR + "/coco/annotations/instances_train2014.json",
    },
    "coco_2014_val": {
        _IM_DIR: _DATA_DIR + "/coco/val2014",
        _ANN_FN: _DATA_DIR + "/coco/annotations/instances_val2014.json",
    },
    "coco_2014_minival": {
        _IM_DIR: _DATA_DIR + "/coco/val2014",
        _ANN_FN: _DATA_DIR + "/coco/annotations/instances_minival2014.json",
    },
    "coco_2014_valminusminival": {
        _IM_DIR: _DATA_DIR + "/coco/val2014",
        _ANN_FN: _DATA_DIR + "/coco/annotations/instances_valminusminival2014.json",
    },
    "coco_2015_test": {
        _IM_DIR: _DATA_DIR + "/coco/test2015",
        _ANN_FN: _DATA_DIR + "/coco/annotations/image_info_test2015.json",
    },
    "coco_2015_test-dev": {
        _IM_DIR: _DATA_DIR + "/coco/test2015",
        _ANN_FN: _DATA_DIR + "/coco/annotations/image_info_test-dev2015.json",
    },
    "coco_2017_test": {  # 2017 test uses 2015 test images
        _IM_DIR: _DATA_DIR + "/coco/test2015",
        _ANN_FN: _DATA_DIR + "/coco/annotations/image_info_test2017.json",
    },
    "coco_2017_test-dev": {  # 2017 test-dev uses 2015 test images
        _IM_DIR: _DATA_DIR + "/coco/test2015",
        _ANN_FN: _DATA_DIR + "/coco/annotations/image_info_test-dev2017.json",
    },
}


def datasets():
    """Retrieve the list of available dataset names."""
    return _DATASETS.keys()


def contains(name):
    """Determine if the dataset is in the catalog."""
    return name in _DATASETS.keys()


def get_image_dir(name):
    """Retrieve the image directory for the dataset."""
    return _DATASETS[name][_IM_DIR]


def get_annotation_filename(name):
    """Retrieve the annotation file for the dataset."""
    return _DATASETS[name][_ANN_FN]

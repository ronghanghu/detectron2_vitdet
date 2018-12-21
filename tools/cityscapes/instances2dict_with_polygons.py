#!/usr/bin/python
#
# Convert instances from png files to a dictionary
# This files is created according to https://github.com/facebookresearch/Detectron/issues/111

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
from PIL import Image

import cv2

# Cityscapes imports
from cityscapesscripts.evaluation.instance import Instance
from cityscapesscripts.helpers.labels import id2label
from cityscapesscripts.helpers.labels import labels

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "helpers")))


def instances2dict_with_polygons(imageFileList, verbose=False):
    imgCount = 0
    instanceDict = {}

    if not isinstance(imageFileList, list):
        imageFileList = [imageFileList]

    if verbose:
        print("Processing {} images...".format(len(imageFileList)))

    for imageFileName in imageFileList:
        # Load image
        img = Image.open(imageFileName)

        # Image as numpy array
        imgNp = np.array(img)

        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp):
            if instanceId < 1000:
                continue
            instanceObj = Instance(imgNp, instanceId)
            instanceObj_dict = instanceObj.toDict()

            # instances[id2label[instanceObj.labelID].name].append(instanceObj.toDict())
            if id2label[instanceObj.labelID].hasInstances:
                mask = (imgNp == instanceId).astype(np.uint8)
                im2, contour, hier = cv2.findContours(
                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )

                polygons = [c.reshape(-1).tolist() for c in contour]
                instanceObj_dict["contours"] = polygons

            instances[id2label[instanceObj.labelID].name].append(instanceObj_dict)

        imgKey = os.path.abspath(imageFileName)
        instanceDict[imgKey] = instances
        imgCount += 1

        if verbose:
            print("\rImages Processed: {}".format(imgCount), end=" ")
            sys.stdout.flush()

    if verbose:
        print("")

    return instanceDict


def main(argv):
    fileList = []
    if len(argv) > 2:
        for arg in argv:
            if "png" in arg:
                fileList.append(arg)
    instances2dict_with_polygons(fileList, True)


if __name__ == "__main__":
    main(sys.argv[1:])
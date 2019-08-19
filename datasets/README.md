For a few datasets that detectron2 natively supports,
it assumes they exist in this directory, with the following directory structure.

You can link the original datasets to this directory.


## Expected dataset structure for COCO instance/keypoint detection:

```
coco/
  annotations/
    instances_train2017.json
    instances_val2017.json
    person_keypoints_train2017.json
    person_keypoints_val2017.json
  train2017/
    # image files that are mentioned in the corresponding json
  val2017/
    # image files that are mentioned in corresponding json
```

You can use the 2014 version of the dataset as well.

Some of the builtin tests (`run_*_tests.sh`) uses a tiny version of the COCO dataset,
which you can download with `./prepare_for_tests.sh`.

## Expected dataset structure for PanopticFPN:
```
coco/
  annotations/
    panoptic_train2017.json
    panoptic_val2017.json
  panoptic_train2017/
    # png annotations
  panoptic_val2017/
    # png annotations
```

Then, run `./prepare_panoptic_fpn.py`, to extract semantic annotations from panoptic annotations.

## Expected dataset structure for cityscapes:
```
cityscapes/
  gtFine/
    train/
      aachen/
        color, instanceIds, labelIds, polygons
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
```

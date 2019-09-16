For a few datasets that detectron2 natively supports,
it assumes they exist in this directory, with the following directory structure.

You can link the original datasets to this directory.


## Expected dataset structure for COCO instance/keypoint detection:

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

You can use the 2014 version of the dataset as well.

Some of the builtin tests (`run_*_tests.sh`) uses a tiny version of the COCO dataset,
which you can download with `./prepare_for_tests.sh`.

## Expected dataset structure for PanopticFPN:
```
coco/
  annotations/
    panoptic_{train,val}2017.json
  panoptic_{train,val}2017/
    # png annotations
```

Then, run `./prepare_panoptic_fpn.py`, to extract semantic annotations from panoptic annotations.

## Expected dataset structure for LVIS instance detection/segmentation:
```
coco/
  {train,val,test}2017/
lvis/
  lvis_v0.5_{train,val}.json
	lvis_v0.5_image_info_test.json
```

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

## Expected dataset structure for Pascal VOC:
```
VOC20{07,12}/
  Annotations/
	ImageSets/
	JPEGImages/
```

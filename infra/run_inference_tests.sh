#!/bin/bash -e

BIN="python tools/train_net.py"
OUTPUT="inference_test_output"
NUM_GPUS=2

for cfg in ./configs/quick_schedules/*inference_acc_test.yaml; do
    if [[ $cfg == *"DC5"* ]]; then
      # TODO
      echo "Skipping $cfg ... See https://github.com/fairinternal/detectron2/issues/114"
      continue
    fi
    echo
    echo "========================================================================"
    echo "Running $cfg ..."
    echo "========================================================================"
    $BIN \
      --eval-only \
      --num-gpus $NUM_GPUS \
      --config-file "$cfg" \
      OUTPUT_DIR $OUTPUT
      rm -rf $OUTPUT
done


echo "========================================================================"
echo "Running demo.py ..."
echo "========================================================================"
DEMO_BIN="python demo/demo.py"
COCO_DIR=datasets/coco/val2014
mkdir -pv $OUTPUT

set -v

$DEMO_BIN --config-file ./configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml \
  --input $COCO_DIR/COCO_val2014_0000001933* --output $OUTPUT
rm -rf $OUTPUT


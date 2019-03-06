#!/bin/bash -e

BIN="python tools/train_net.py"
OUTPUT="instant_test_output"
NUM_GPUS=2

for cfg in ./configs/quick_schedules/*instant_test.yaml; do
    echo "========================================================================"
    echo "Running $cfg ..."
    echo "========================================================================"
    $BIN --num-gpus $NUM_GPUS --config-file "$cfg" \
      SOLVER.IMS_PER_BATCH $(($NUM_GPUS * 2)) \
      OUTPUT_DIR "$OUTPUT"
    rm -rf "$OUTPUT"
done


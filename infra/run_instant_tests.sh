#!/bin/bash -e

BIN="python tools/train_net.py"
OUTPUT="instant_test_output"

for cfg in ./configs/quick_schedules/*instant_test.yaml; do
    echo "========================================================================"
    echo "Running $cfg ..."
    echo "========================================================================"
    $BIN --num-gpus 2 --config-file "$cfg" \
        SOLVER.IMS_PER_BATCH 4 \
        OUTPUT_DIR "$OUTPUT"
    rm -rf "$OUTPUT"
done


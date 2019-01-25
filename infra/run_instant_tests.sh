#!/bin/bash -e

BIN="python tools/train_net.py"
OUTPUT="instant_test_output"

for cfg in ./configs/quick_schedules/*instant_test.yaml; do
        echo "Running "$cfg" ..."
        $BIN --num-gpus 2 --config-file "$cfg" \
                SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 2 \
                OUTPUT_DIR "$OUTPUT"
        rm -rf "$OUTPUT"
done


#!/bin/bash -e

# Example of launching a two node job:
#
# srun -o /path/to/logs/%j.%t.%N.out -e /path/to/logs/%j.%t.%N.err \
#    --gres=gpu:8 --mem 200GB -N 2 -p uninterrupted -t 2880 -J TwoNodeFPN2x \
#   ./infra/fair/launch_distributed.sh --config-file configs/e2e_mask_rcnn_R_50_FPN_1x.yaml \
#   SOLVER.IMS_PER_BATCH 32 SOLVER.BASE_LR 0.04
#
# Note that the above command changes the batch size & learning rate, but not the training
# iterations, so it effectively becomes FPN2x, not 1x.

printenv | grep SLURM

MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
DIST_URL="tcp://$MASTER_NODE:12389"
NUM_GPUS_PER_NODE=$(echo "$SLURM_STEP_GRES" | tr ',' '\n' | wc -l)

python -u tools/train_net.py \
	--num-gpus "$NUM_GPUS_PER_NODE" \
	--num-machines "$SLURM_NNODES" --machine-rank "$SLURM_NODEID" \
	--dist-url "$DIST_URL" "$@"

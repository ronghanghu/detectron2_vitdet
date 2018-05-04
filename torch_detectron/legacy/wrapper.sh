#!/bin/bash

echo Running Job $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES

#JOBNAME="detectron_faster_rcnn_R50_"$SLURM_JOB_ID
JOBNAME="detectron_fpn_R50_"$SLURM_JOB_ID
#JOBNAME=${JOBNAME}_correct_momentum
#_pinmemory_false
#_time_batch_before_log
#_debug
# _no_bp_through_roi

SAVEPATH="/checkpoint02/fmassa/"$JOBNAME

echo Saving to $SAVEPATH

DETECTRON_DIR="/private/home/fmassa/github/detectron.pytorch/torch_detectron/"

ARGS=" --epochs 2 --lr-steps 1 --imgs-per-batch 2"
# ARGS=" --lr 0.02 --imgs-per-batch 2 --epochs 12 --lr-steps 7 11 --num-workers 2 --no-ar-group"
ARGS=" --lr 0.015 --imgs-per-batch 2 --epochs 12 --lr-steps 7 11 --num-workers 4"
#ARGS=" --lr 0.005 --imgs-per-batch 2 --epochs 12 --lr-steps 7 11 --num-workers 0 --checkpoint /checkpoint02/fmassa/detectron_fpn_R50_3113002/model_10.pth"
ARGS=" --lr 0.005 --imgs-per-batch 2 --epochs 12 --lr-steps 7 11 --num-workers 0 --checkpoint /checkpoint02/fmassa/detectron_fpn_R50_3153229/model_11.pth"
#ARGS=" --lr 0.01 --imgs-per-batch 1 --epochs 12 --lr-steps 7 11 --num-workers 1 --no-ar-group --checkpoint /checkpoint02/fmassa/detectron_faster_rcnn_R50_2571743_try_sleep/model.pth"

NUM_GPUS=2

rm -rf /tmp/torch_extensions/detectron_modules

echo Compiling extension
/private/home/fmassa/.conda/envs/detectron_v2/bin/python -c "from lib.layers import _C"

echo Run code

# /private/home/fmassa/.conda/envs/detectron_v2/bin/python -m torch.distributed.launch --nproc_per_node=8 training_faster_rcnn.py --save $SAVEPATH
OMP_NUM_THREADS=1 NCCL_DEBUG=INFO /private/home/fmassa/.conda/envs/detectron_v2/bin/python ${DETECTRON_DIR}launch_distributed.py --nproc_per_node=${NUM_GPUS} ${DETECTRON_DIR}training_fpn.py --save $SAVEPATH --dataset coco  $ARGS

# /private/home/fmassa/.conda/envs/detectron_v2/bin/python training_faster_rcnn.py

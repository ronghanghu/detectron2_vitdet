#!/bin/bash

echo Running Job $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES

# if running from slurm
if [ ! -z ${SLURM_JOB_ID+x} ]; then
  # JOBNAME="rpn_R50_"$SLURM_JOB_ID
  JOBNAME="fpn_R50_"$SLURM_JOB_ID
  echo Job Name $JOBNAME

  export SAVE_DIR="/checkpoint02/fmassa/detectron_logs/"$JOBNAME
fi

echo Saving to $SAVE_DIR

DETECTRON_DIR="/private/home/fmassa/github/detectron.pytorch/torch_detectron/"

NUM_GPUS=`python -c "print(len('${CUDA_VISIBLE_DEVICES}'.split(',')))"`

echo Using $NUM_GPUS GPUs


ARGS="$@"

if [ -z "$ARGS" ]; then
  # ARGS="--config-file /private/home/fmassa/github/detectron.pytorch/configs/mask_faster_rcnn_r50.py"
  # ARGS="--config-file /private/home/fmassa/github/detectron.pytorch/configs/rpn_fpn_r50.py"
  # ARGS="--config-file /private/home/fmassa/github/detectron.pytorch/configs/fpn_r50.py"
  # ARGS="--config-file /private/home/fmassa/github/detectron.pytorch/configs/mask_fpn_r50.py"
  # ARGS="--config-file /private/home/fmassa/github/detectron.pytorch/configs/rpn_r50.py"
  # ARGS="--config-file /private/home/fmassa/github/detectron.pytorch/configs/fpn_r50_adaptive_upsample.py"
  ARGS="--config-file /private/home/fmassa/github/detectron.pytorch/configs/keypoint_fpn_r50.py"
fi

PYTHON_INTER="/private/home/fmassa/.conda/envs/detectron_v2/bin/python"
# PYTHON_INTER="/private/home/fmassa/.conda/envs/detectron_cuda8/bin/python"

# export CHECKPOINT_FILE="/checkpoint02/fmassa/detectron_logs/faster_rcnn_R50_3970040/model_161249.pth"

echo Launching command
#PYTHONUNBUFFERED=True NCCL_DEBUG=INFO $PYTHON_INTER -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ${DETECTRON_DIR}train.py $@
#PYTHONUNBUFFERED=True NCCL_DEBUG=INFO $PYTHON_INTER -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ${DETECTRON_DIR}train.py --config-file "/private/home/fmassa/github/detectron.pytorch/configs/faster_rcnn_r50.py"
#PYTHONUNBUFFERED=True NCCL_DEBUG=INFO $PYTHON_INTER -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ${DETECTRON_DIR}train.py --config-file "/private/home/fmassa/github/detectron.pytorch/configs/faster_rcnn_r50_relaunch.py"
# $PYTHON_INTER ${DETECTRON_DIR}train.py

# PYTHONUNBUFFERED=True NCCL_DEBUG=INFO $PYTHON_INTER -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ${DETECTRON_DIR}train.py --config-file "/private/home/fmassa/github/detectron.pytorch/configs/fpn_r50.py"
#PYTHONUNBUFFERED=True NCCL_DEBUG=INFO $PYTHON_INTER -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ${DETECTRON_DIR}train.py --config-file "/private/home/fmassa/github/detectron.pytorch/configs/rpn_r50_quick_new.py"

#PYTHONUNBUFFERED=True NCCL_DEBUG=INFO $PYTHON_INTER -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ${DETECTRON_DIR}train.py --config-file "/private/home/fmassa/github/detectron.pytorch/configs/mask_faster_rcnn_r50.py"
# PYTHONUNBUFFERED=True NCCL_DEBUG=INFO $PYTHON_INTER -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ${DETECTRON_DIR}train.py $ARGS

PYTHONUNBUFFERED=True NCCL_DEBUG=INFO $PYTHON_INTER -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ${DETECTRON_DIR}train.py $@

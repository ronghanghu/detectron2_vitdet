#!/bin/bash

echo Running Job $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES

# if running from slurm
if [ ! -z ${SLURM_JOB_ID+x} ]; then
  JOBNAME="fpn_R50_"$SLURM_JOB_ID
  echo Job Name $JOBNAME

  export SAVE_DIR="/checkpoint02/fmassa/detectron_logs/"$JOBNAME
fi

echo Saving to $SAVE_DIR

DETECTRON_DIR="/private/home/fmassa/github/detectron.pytorch/torch_detectron/"

# count number of GPUs that are available
# don't know bash, so use python instead :-)
NUM_GPUS=`python -c "print(len('${CUDA_VISIBLE_DEVICES}'.split(',')))"`
echo Using $NUM_GPUS GPUs

ARGS="$@"

if [ -z "$ARGS" ]; then
  ARGS="--config-file /private/home/fmassa/github/detectron.pytorch/configs/keypoint_fpn_r50.py"
fi

PYTHON_INTER="/private/home/fmassa/.conda/envs/detectron_v2/bin/python"

echo Launching command
PYTHONUNBUFFERED=True NCCL_DEBUG=INFO $PYTHON_INTER -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ${DETECTRON_DIR}train.py $ARGS

#!/bin/bash

echo Running Job $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES

# if running from slurm
# set up the directory where to save the data
if [ ! -z ${SLURM_JOB_ID+x} ]; then
  JOBNAME="detectron_"$SLURM_JOB_ID
  echo Job Name $JOBNAME

  export SAVE_DIR="/checkpoint02/$(whoami)/detectron_logs/"$JOBNAME
fi

echo Saving to $SAVE_DIR

# count number of GPUs that are available
# don't know bash, so use python instead :-)
NUM_GPUS=`python -c "print(len('${CUDA_VISIBLE_DEVICES}'.split(',')))"`
echo Using $NUM_GPUS GPUs

echo Launching command
PYTHONUNBUFFERED=True NCCL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} train.py $@

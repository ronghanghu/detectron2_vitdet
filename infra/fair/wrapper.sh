#!/bin/bash

# This script needs to be run from the base folder

echo Running Job $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES

# if running from slurm
# set up the directory where to save the data
if [ ! -z ${SLURM_JOB_ID+x} ]; then
  JOBNAME="detectron_"$SLURM_JOB_ID
  echo Job Name $JOBNAME

  OUTPUT_DIR="/checkpoint02/$(whoami)/detectron_logs/"$JOBNAME
  OUTPUT_DIR_CMD="OUTPUT_DIR "$OUTPUT_DIR
fi

echo Saving to $OUTPUT_DIR

# count number of GPUs that are available
# don't know bash, so use python instead :-)
NUM_GPUS=`python -c "print(len('${CUDA_VISIBLE_DEVICES}'.split(',')))"`
echo Using $NUM_GPUS GPUs

echo Launching command

echo "PYTHONUNBUFFERED=True NCCL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} tools/train_net.py $@ ${OUTPUT_DIR_CMD}"

PYTHONUNBUFFERED=True NCCL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} tools/train_net.py $@ ${OUTPUT_DIR_CMD}


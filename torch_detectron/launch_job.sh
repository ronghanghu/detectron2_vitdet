#!/bin/bash

#SBATCH --job-name=detectron

#SBATCH --output=/checkpoint/%u/jobs/detectron-keypoint-%j.out

#SBATCH --error=/checkpoint/%u/jobs/detectron-keypoint-%j.err

#SBATCH --partition=uninterrupted

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=80

#SBATCH --gres=gpu:volta:8

#SBATCH --time=72:00:00

module purge

module load anaconda3/5.0.1
#module load cuda/8.0
module load cuda/9.0
#module load cudnn/v6.0
module load cudnn/v7.0-cuda.9.0
#module load NCCL/2.0.5
module load NCCL/2.2.13-cuda.9.0

source activate detectron_v2
#source activate detectron_cuda8

#srun env
# srun --label wrapper.sh
srun --label wrapper.sh $@

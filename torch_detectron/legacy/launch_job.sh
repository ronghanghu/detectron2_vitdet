#!/bin/bash

#SBATCH --job-name=detectron

#SBATCH --output=/checkpoint/%u/jobs/detectron-%j.out

#SBATCH --error=/checkpoint/%u/jobs/detectron-%j.err

#SBATCH --partition=learnfair

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=20

#SBATCH --gres=gpu:2

#SBATCH --time=72:00:00

module purge

module load anaconda3/5.0.1
module load cuda/9.0
module load cudnn/v7.0-cuda.9.0
module load NCCL/2.1.15-1-cuda.9.0

source activate detectron_v2

#srun env
srun --label wrapper.sh


#!/bin/bash

#SBATCH --job-name=detectron

#SBATCH --output=/checkpoint/%u/jobs/detectron-%j.out

#SBATCH --error=/checkpoint/%u/jobs/detectron-%j.err

#SBATCH --partition=uninterrupted

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=80

#SBATCH --mem=128G

#SBATCH --gres=gpu:8

#SBATCH --time=72:00:00

#srun env
srun --label wrapper.sh $@

#!/bin/bash

#SBATCH --job-name=train_fast
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gpus=2080ti:1
#SBATCH --time=72:00:00
#SBATCH -o /gscratch/weirdlab/ecai0608/slurm_logs/%x_%j.out

# Grabbing arguments.
CONTAINER_PATH=/gscratch/weirdlab/ecai0608/fast_project/fast.sif
WANDB_MODE=$1
COMMAND=$2

# Run the training script. NOTE: this should be run from the root directory.
./scripts/run.sh $CONTAINER_PATH $WANDB_MODE $COMMAND
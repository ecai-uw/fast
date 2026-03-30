#!/bin/bash

# Simple script to copy a WandB run log directory from a remote cluster to local machine.
# General usage:
# bash scripts/copy_run.sh [CLUSTER] [WANDB_ID]

CLUSTER=$1
WANDB_ID=$2

if [ "$CLUSTER" == "klone" ]; then
    LOG_PATH="/gscratch/scrubbed/ecai0608/fast_project/fast/logs"
elif [ "$CLUSTER" == "tillicum" ]; then
    LOG_PATH="/gpfs/scrubbed/ecai0608/fast_project/fast/logs"
else
    echo "Unknown cluster: $CLUSTER"
    exit 1
fi

# Copy the WandB log directory, without replay buffer.
# Also exclude checkpoints, except for final.zip.

# rsync -avP --exclude="*replay_buffer*" --exclude="*media*" ecai0608@${CLUSTER}.hyak.uw.edu:${LOG_PATH}/${WANDB_ID}/ ./logs/${WANDB_ID}/
rsync -avP --exclude="*replay_buffer*" --exclude="*media*" --include="*checkpoint/final.zip" --exclude="*checkpoint/*" ecai0608@${CLUSTER}.hyak.uw.edu:${LOG_PATH}/${WANDB_ID}/ ./logs/${WANDB_ID}/
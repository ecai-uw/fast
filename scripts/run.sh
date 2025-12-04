#!/bin/bash

# General usage:
# bash scripts/run.sh [CONTAINER_PATH] [WANDB_MODE] [ARGS]

CONTAINER_PATH=$1
WANDB_MODE=$2
COMMAND=$3

apptainer exec --nv --writable-tmpfs \
    --containall --no-home \
    --home /root \
    --bind $(pwd):/opt/code/fast/ \
    --env WANDB_MODE=$WANDB_MODE \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --pwd /opt/code/fast/ \
    $CONTAINER_PATH \
    python train_fast.py \
    $COMMAND

# Some notes for the command above:
# --nv:
#       enables NVIDIA GPU support inside the container.
#
# --writable-tmpfs:
#       allows writing to the container filesystem by creating a temporary writable layer (useful for many 
#       packages/generic logging utilities).
#
# --containall --no-home:
#       ensures a clean environment without access to host user files or environment variables.
#
# --home /root:
#       sets the container home directory to /root (instead of host $HOME).
#
# --bind $(pwd):/opt/code/fast:
#       requires that current working directory be the project root.

# --env WANDB_API_KEY: 
#       requires that the WANDB_API_KEY environment variable is set in the host system.
#
# --pwd /opt/code/fast/: 
#       sets the container working directory to the project root.
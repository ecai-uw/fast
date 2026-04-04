#!/bin/bash

# General usage:
# bash scripts/eval.sh [CONTAINER_PATH] [WANDB_MODE] [TASK] [ARGS]

# method type

CONTAINER_PATH=$1
WANDB_MODE=$2
TASK=$3
shift
shift
shift
COMMAND=$@

# Parsing task type to set config file.
if [ "$TASK" == "robomimic_lift" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_lift.yaml"
elif [ "$TASK" == "robomimic_can" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_can.yaml"
elif [ "$TASK" == "robomimic_square" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_square.yaml"
elif [ "$TASK" == "robomimic_transport" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_transport.yaml"
# Image-based tasks.
elif [ "$TASK" == "robomimic_lift_img" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_lift_img.yaml"
elif [ "$TASK" == "robomimic_can_img" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_can_img.yaml"
elif [ "$TASK" == "robomimic_square_img" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_square_img.yaml"
elif [ "$TASK" == "robomimic_transport_img" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_transport_img.yaml"
# Image-based multi-view tasks.
elif [ "$TASK" == "robomimic_lift_img_mv" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_lift_img_mv.yaml"
elif [ "$TASK" == "robomimic_can_img_mv" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_can_img_mv.yaml"
elif [ "$TASK" == "robomimic_square_img_mv" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_square_img_mv.yaml"
elif [ "$TASK" == "robomimic_transport_img_mv" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_transport_img_mv.yaml"
# Image-based multi-view tasks with depth.
elif [ "$TASK" == "robomimic_lift_img_depth_mv" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_lift_img_depth_mv.yaml"
elif [ "$TASK" == "robomimic_can_img_depth_mv" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_can_img_depth_mv.yaml"
elif [ "$TASK" == "robomimic_square_img_depth_mv" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_square_img_depth_mv.yaml"
elif [ "$TASK" == "robomimic_transport_img_depth_mv" ]; then
    COMMAND_PREFIX="--config-path=cfg/robomimic --config-name=fast_transport_img_depth_mv.yaml"
else
    echo "Unknown task type: $TASK"
    exit 1
fi
COMMAND="$COMMAND_PREFIX $COMMAND"

# apptainer shell --nv --writable-tmpfs \
#     --containall --no-home \
#     --home /root \
#     --bind $(pwd):/opt/code/fast/ \
#     --env WANDB_MODE=$WANDB_MODE \
#     --env WANDB_API_KEY=$WANDB_API_KEY \
#     --pwd /opt/code/fast/ \
#     $CONTAINER_PATH \
#     # python eval_fast.py \
#     # $COMMAND

apptainer exec --nv --writable-tmpfs \
    --containall --no-home \
    --home /root \
    --bind $(pwd):/opt/code/fast/ \
    --env WANDB_MODE=$WANDB_MODE \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --pwd /opt/code/fast/ \
    $CONTAINER_PATH \
    python eval_fast.py \
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
# TODO: this can be modified so that the script can be run anywhere.
# --bind $(pwd):/opt/code/fast:
#       requires that current working directory be the project root.

# --env WANDB_API_KEY: 
#       requires that the WANDB_API_KEY environment variable is set in the host system.
#
# --pwd /opt/code/fast/: 
#       sets the container working directory to the project root.
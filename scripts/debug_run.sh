#!/bin/bash

# General usage:
# bash scripts/debug_run.sh [CONTAINER_PATH] [WANDB_MODE] [ARGS]

# TODO: more args to add in the future: task type (this will determine configs)
# method type

METHOD=$1
CONTAINER_PATH=~/containers/fast.sif
shift
COMMAND=$@


if [ "$METHOD" == "local" ]; then

    WANDB_MODE=disabled python train_fast.py \
        num_evals=2 env.n_envs=1 env.n_eval_envs=1 \
        train.init_rollout_steps=20 \
        base.fqe_steps=50 base.vd_steps=50 \
        total_timesteps=10 \
        $COMMAND

elif [ "$METHOD" == "local-container" ]; then

    apptainer exec --nv --writable-tmpfs \
        --containall --no-home \
        --home /root \
        --bind $(pwd):/opt/code/fast/ \
        --env WANDB_MODE=disabled \
        --env WANDB_API_KEY=$WANDB_API_KEY \
        --pwd /opt/code/fast/ \
        $CONTAINER_PATH \
        python train_fast.py \
        num_evals=2 env.n_envs=1 env.n_eval_envs=1 \
        train.init_rollout_steps=20 \
        base.fqe_steps=50 base.vd_steps=50 \
        total_timesteps=10 \
        $COMMAND

else
    echo "Unknown method: $METHOD"
    exit 1
fi
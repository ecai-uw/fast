#!/bin/bash

# General usage:
# bash scripts/debug_eval.sh [CONTAINER_PATH] [WANDB_MODE] [ARGS]

# TODO: more args to add in the future: task type (this will determine configs)
# method type

# Usage: scripts/debug_eval.sh [method] [task] [additional args]

CONTAINER_PATH=~/containers/fast_24.04.sif
WANDB_MODE=disabled

METHOD=$1
TASK=$2
shift
shift
COMMAND=$@


if [ "$METHOD" == "local" ]; then

    WANDB_MODE=$WANDB_MODE python train_fast.py \
        num_evals=5 env.n_envs=1 env.n_eval_envs=5 \
        train.init_rollout_steps=20 \
        base.fqe_steps=50 base.vd_steps=50 \
        total_timesteps=10 \
        $COMMAND

elif [ "$METHOD" == "local-container" ]; then

    ./scripts/eval.sh $CONTAINER_PATH $WANDB_MODE $TASK \
        num_evals=200 env.n_envs=1 env.n_eval_envs=25 deterministic_eval=True \
        $COMMAND

else
    echo "Unknown method: $METHOD"
    exit 1
fi
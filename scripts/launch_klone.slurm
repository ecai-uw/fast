#!/bin/bash

#SBATCH --job-name=train_fast
#SBATCH --partition=ckpt
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --array=1-3
#SBATCH --requeue
#SBATCH -o /gscratch/scrubbed/ecai0608/fast_project/fast/logs/slurm_logs/%x_%A/seed%a/job.out
#SBATCH -e /gscratch/scrubbed/ecai0608/fast_project/fast/logs/slurm_logs/%x_%A/seed%a/job.err

# Sample usage:
# sbatch scripts/launch_klone.slurm online robomimic_can policy.type=residual policy.shape_rewards=False

# Setting container path and parsing slurm log directory.
CONTAINER_PATH="/gscratch/scrubbed/ecai0608/fast_project/fast.sif"
SLURM_LOG_DIR="slurm_logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/seed${SLURM_ARRAY_TASK_ID}"
WANDB_MODE=$1
TASK=$2
shift
shift
COMMAND=$@

# Adding seed config to the command.
COMMAND_PREFIX=" seed=$SLURM_ARRAY_TASK_ID"

# Handling slurm restart logic for pre-empted jobs.
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
if [ $RESTART_COUNT -gt 0 ]; then
    # Grabbing WandB run ID to resume training.
    WANDB_ID=$(< ./logs/$SLURM_LOG_DIR/wandb_run_id.txt)
    COMMAND_PREFIX+=" resume=True wandb.id=$WANDB_ID"
else
    # First attempt, no need to resume.
    COMMAND_PREFIX+=" resume=False wandb.save_id_local_path=$SLURM_LOG_DIR/wandb_run_id.txt"
fi

# Logging restart count.
echo "SLURM_RESTART_COUNT: $RESTART_COUNT"

# Placing prefix commands before the user command to allow for overrides.
COMMAND="$COMMAND_PREFIX $COMMAND"
# Run the training script. NOTE: this should be run from the project root.
./scripts/run.sh $CONTAINER_PATH $WANDB_MODE $TASK $COMMAND
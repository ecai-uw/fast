#!/bin/bash

# Simple script to copy local code to remote clusters (Klone and Tillicum).
# General usage:
# bash scripts/copy_code.sh [TAG] [CLUSTER]

TAG=$1
CLUSTER=$2

if [ "$CLUSTER" == "klone" ]; then
    # CODE_PATH="/gscratch/weirdlab/ecai0608/fast_project/fast/"
    CODE_PATH="/gscratch/scrubbed/ecai0608/fast_project/fast/"
elif [ "$CLUSTER" == "tillicum" ]; then
    CODE_PATH="/gpfs/scrubbed/ecai0608/fast_project/fast/"
else
    echo "Unknown cluster: $CLUSTER"
    exit 1
fi

# Copy the code directory, excluding large data files and logs.
rsync $TAG --exclude="wandb/*" --exclude="logs/*" --exclude=".git/*" --exclude="debug/*" ~/fast/ ecai0608@${CLUSTER}.hyak.uw.edu:${CODE_PATH}
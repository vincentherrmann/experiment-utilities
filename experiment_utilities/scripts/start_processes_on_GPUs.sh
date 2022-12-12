#!/bin/bash

# This script was written mostly by ChatGPT. It is used to start multiple processes on multiple GPUs.
# Its main purpose is starting the agents of a wandb sweep. To run it, execute
# bash start_processes_on_GPUs.sh <number of processes> <list of available GPUs> <command to run>
# so e.g.: start_processes_on_GPUs.sh 8 0 1 2 3 "wandb agent vincentherrmann/some_project/jsyfkhzh"

# Set default values for the number of agents and GPUs
num_agents=${1:-1}

# get the list of GPUs (so the second argument and all following arguments except the last one)
# gpus=${@:2:$(($#-2))}

length=$(($#-1))
gpus=${@:1:$length}

#gpus=(${@:2:$num_agents})

#gpus=${2:-0}
num_gpus=${#gpus}

echo "gpus: $gpus"
echo "num available gpus: $num_gpus"

# Get the command to run from the remaining command line arguments
shift $(($# - 1))
command=$1

# Start Wandb agents
for i in $(seq 1 $num_agents)
do
    # Assign each agent to a different GPU (cycling through the list of GPUs if necessary)
    gpu=${gpus[$((i % $num_gpus))]}

    # Start the agent with the specified command with the CUDA_VISIBLE_DEVICES environment variable set
    echo "command to execute: CUDA_VISIBLE_DEVICES=$gpu $command"
    CUDA_VISIBLE_DEVICES=$gpu $command &
done
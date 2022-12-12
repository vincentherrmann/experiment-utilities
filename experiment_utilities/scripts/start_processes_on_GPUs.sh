#!/bin/bash

# This script was written mostly by ChatGPT. It is used to start multiple processes on multiple GPUs.
# Its main purpose is starting the agents of a wandb sweep. To run it, execute
# bash start_processes_on_GPUs.sh <number of processes> <list of available GPUs> <command to run>
# so e.g.: start_processes_on_GPUs.sh 8 0 1 2 3 "wandb agent vincentherrmann/some_project/jsyfkhzh"

# Set default values for the number of agents and GPUs
num_agents=${1:-1}
gpus=${2:-0}

# Get the command to run from the remaining command line arguments
shift $(($# - 1))
command=$1

# Start Wandb agents
for i in $(seq 1 $num_agents)
do
    # Assign each agent to a different GPU (cycling through the list of GPUs if necessary)
    gpu=${gpus[$i % $num_gpus]}

    # Start the agent with the specified command with the CUDA_VISIBLE_DEVICES environment variable set
    CUDA_VISIBLE_DEVICES=$gpu $command &
done
#!/bin/bash

# List Wandb agents
# ps aux | grep "wandb agent"

# Stop Wandb agents
for i in {1..5}
do
    kill $(pgrep -f "wandb agent")
done
#!/usr/bin/env bash

config="experiments/config/d4rl/gym_rvs_r.cfg"
declare -a envs=("halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-expert-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
seeds=5
use_gpu=true

for env in "${envs[@]}"; do
  for seed in $(seq 0 $((seeds-1))); do
    if [ "$use_gpu" = true ]; then
      python src/rvs/train.py --configs "$config" --env_name "$env" --seed "$seed" --use_gpu
    else
      python src/rvs/train.py --configs "$config" --env_name "$env" --seed "$seed"
    fi
  done
done

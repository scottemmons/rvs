#!/usr/bin/env bash

config="experiments/config/d4rl/antmaze_rvs_g.cfg"
declare -a envs=("antmaze-umaze-v0" "antmaze-umaze-diverse-v0" "antmaze-medium-diverse-v0" "antmaze-medium-play-v0" "antmaze-large-diverse-v0" "antmaze-large-play-v0")
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

#!/usr/bin/env bash

configs=(experiments/config/gcsl/*_rvs_g.cfg)
seeds=5
use_gpu=true

for config in "${configs[@]}"; do
  for seed in $(seq 0 $((seeds-1))); do
    if [ "$use_gpu" = true ]; then
      python src/rvs/train.py --configs "$config" --seed "$seed" --use_gpu
    else
      python src/rvs/train.py --configs "$config" --seed "$seed"
    fi
  done
done

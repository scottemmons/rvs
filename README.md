[![CircleCI](https://circleci.com/gh/scottemmons/rvs/tree/main.svg?style=svg)](https://circleci.com/gh/scottemmons/rvs/tree/main)
[![codecov](https://codecov.io/gh/scottemmons/rvs/branch/main/graph/badge.svg)](https://codecov.io/gh/scottemmons/rvs)

# Reinforcement Learning via Supervised Learning

# Installation

Run
```bash
pip install -e .
```

The code depends on MuJoCo 2.0 and MuJoCo 2.1. Here are [instructions for installing MuJoCo 2.0](https://github.com/openai/mujoco-py/tree/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb#install-mujoco)
and [instructions for installing MuJoCo 2.1](https://github.com/openai/mujoco-py/tree/fb4babe73b1ef18b4bea4c6f36f6307e06335a2f#install-mujoco).

If you use the provided `Dockerfile`, it will automatically handle the MuJoCo
dependencies for you. For example:
```bash
docker build -t rvs:latest .
docker run -it --rm -v $(pwd):/rvs rvs:latest bash
cd rvs
pip install -e .
```

# Reproducing Experiments

The `experiments` directory contains a launch script for each environment suite. For
example, to reproduce the RvS-R results in D4RL Gym locomotion, run
```bash
bash experiments/launch_gym_rvs_r.sh
```
Each launch script corresponds to a configuration file in `experiments/config` which
serves as a reference for the hyperparameters associated with each experiment.

# Adding New Environments

To run RvS on an environment of your own, you need to create a suitable dataset class.
For example, in `src/rvs/dataset.py`, we have a dataset class for the GCSL environments,
a dataset class for RvS-R in D4RL, and a dataset class for RvS-G in D4RL. In particular,
the `D4RLRvSGDataModule` allows for conditioning on arbitrary dimensions of the goal
state using the `goal_columns` attribute; for AntMaze, we set `goal_columns` to `(0, 1)`
to condition only and the x and y coordinates of the goal state.
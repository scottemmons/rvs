[![CircleCI](https://circleci.com/gh/scottemmons/rvs/tree/main.svg?style=svg)](https://circleci.com/gh/scottemmons/rvs/tree/main)
[![codecov](https://codecov.io/gh/scottemmons/rvs/branch/main/graph/badge.svg)](https://codecov.io/gh/scottemmons/rvs)
[![arXiv](https://img.shields.io/badge/arXiv-2112.10751-b31b1b.svg)](https://arxiv.org/abs/2112.10751)

# Reinforcement Learning via Supervised Learning

# Installation

Run
```bash
pip install -e .
```
in an environment with Python >= 3.7.0, <3.9.

The code depends on MuJoCo 2.1.0 (for mujoco-py) and MuJoCo 2.1.1 (for dm-control). Here are [instructions for installing MuJoCo 2.1.0](https://github.com/openai/mujoco-py/tree/fb4babe73b1ef18b4bea4c6f36f6307e06335a2f#install-mujoco)
and [instructions for installing MuJoCo 2.1.1](https://github.com/deepmind/dm_control/tree/84fc2faa95ca2b354f3274bb3f3e0d29df7fb337#requirements-and-installation).

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
to condition only on the x and y coordinates of the goal state.

# Baseline Numbers

We replicated CQL using [this codebase](https://github.com/scottemmons/youngs-cql),
which was recommended to us by the CQL authors. All hyperparameters and logs from our
replication runs can be viewed at our [CQL-R Weights & Biases project](https://wandb.ai/scottemmons/SimpleSAC--cql).

We replicated Decision Transformer using [our fork](https://github.com/scottemmons/decision-transformer)
of the author's codebase, which we customized to add AntMaze. All hyperparameters and
logs from our replication runs can be viewed at our [DT Weights & Biases project](https://wandb.ai/scottemmons/decision-transformer).

# Citing RvS

To cite RvS, you can use the following BibTeX entry:

```bibtex
@inproceedings{emmons2022rvs,
    title={RvS: What is Essential for Offline {RL} via Supervised Learning?},
    author={Scott Emmons and Benjamin Eysenbach and Ilya Kostrikov and Sergey Levine},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=S874XAIpkR-}
}
```

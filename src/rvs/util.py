"""Utilities used throughout the codebase."""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Dict, List, Optional, Tuple, Union

from d4rl import offline_env
from gym import spaces
import numpy as np
import torch
import wandb
from wandb.sdk.wandb_run import Run

from rvs import step, train


def configure_gpu(use_gpu: bool, which_gpu: int) -> torch.device:
    """Set the GPU to be used for training."""
    if use_gpu:
        device = torch.device("cuda")
        # Only occupy one GPU, as in https://stackoverflow.com/questions/37893755/
        # tensorflow-set-cuda-visible-devices-within-jupyter
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu)
    else:
        device = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    return device


def set_seed(seed: Optional[int]) -> None:
    """Set the numpy, random, and torch random seeds."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)


def extract_traj_markers(
    dataset: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a D4RL dataset, return starts, ends, and lengths of trajectories."""
    dones = np.logical_or(dataset["terminals"], dataset["timeouts"])
    return extract_done_markers(dones)


def extract_done_markers(
    dones: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a per-timestep dones vector, return starts, ends, and lengths of trajs."""
    (ends,) = np.where(dones)
    starts = np.concatenate(([0], ends[:-1] + 1))
    lengths = ends - starts + 1

    return starts, ends, lengths


def collect_timestep_indices(
    dones: np.ndarray,
    trajectory_indices: Union[List[int], np.ndarray],
) -> np.ndarray:
    """Find all timestep indices within the given trajectory indices."""
    starts, ends, _ = extract_done_markers(dones)
    starts = starts[trajectory_indices]
    ends = ends[trajectory_indices]

    timestep_indices = []
    for start, end in zip(starts, ends):
        timesteps = np.arange(start, end + 1)
        timestep_indices.append(timesteps)
    timestep_indices = (
        np.concatenate(timestep_indices) if len(timestep_indices) > 0 else np.array([])
    )

    return timestep_indices


def concatenate_two_boxes(box_a: spaces.Box, box_b: spaces.Box) -> spaces.Box:
    """Concatenate two Box spaces into one Box space."""
    if not isinstance(box_a, spaces.Box) or not isinstance(box_b, spaces.Box):
        raise ValueError("This method will only concatenate Box spaces")

    lows = np.concatenate([box_a.low, box_b.low])
    highs = np.concatenate([box_a.high, box_b.high])
    dtype = np.result_type(*[box_a.dtype, box_b.dtype])

    return spaces.Box(low=lows, high=highs, dtype=dtype)


def duplicate_observation_space(observation_space: spaces.Box) -> spaces.Box:
    """Double the observation spaces by concatenating it with itself."""
    if not isinstance(observation_space, spaces.Box):
        raise ValueError("This method will only duplicate Box observation_spaces")

    return concatenate_two_boxes(observation_space, observation_space)


def flatten_observation_goal_spaces(observation_space: spaces.Dict) -> spaces.Box:
    """Create a Box space out of the observation and desired_goal of the Dict space."""
    if not isinstance(observation_space, spaces.Dict):
        raise ValueError("This method will only flatten Dict observation_spaces")

    return concatenate_two_boxes(
        observation_space["observation"],
        observation_space["desired_goal"],
    )


def create_observation_goal_space(
    observation_space: Union[spaces.Box, spaces.Dict],
) -> spaces.Box:
    """Take the observation space and produce a space with observations and goals."""
    if isinstance(observation_space, spaces.Dict):
        return flatten_observation_goal_spaces(observation_space)
    else:
        return duplicate_observation_space(observation_space)


def add_scalar_to_space(
    observation_space: Union[spaces.Box, spaces.Dict],
) -> spaces.Box:
    """Add one scalar to the observation space."""
    if isinstance(observation_space, spaces.Dict):
        observation_space = observation_space["observation"]
    if not isinstance(observation_space, spaces.Box):
        raise ValueError("This method can only add reward to a Box observation_space")

    lows = np.concatenate([observation_space.low, [-np.inf]])
    highs = np.concatenate([observation_space.high, [np.inf]])
    return spaces.Box(low=lows, high=highs, dtype=observation_space.dtype)


def resolve_out_directory(run_id: str, entity: str) -> Tuple[str, Run]:
    """Download wandb run and return its local output directory."""
    # get the wandb run from the api
    api = wandb.Api()
    api_run = api.run(f"{entity}/{train.wandb_project}/{run_id}")

    # resume the wandb run
    wandb.init(
        entity=entity,
        project=train.wandb_project,
        id=run_id,
        resume="must",
    )
    wandb_run = wandb.run

    print("Downloading files from wandb...")
    for file in api_run.files():
        wandb_run.restore(file.name)
    print("Successfully downloaded files")

    return wandb_run.dir, wandb_run


def sorted_glob(*args, **kwargs) -> List[str]:
    """A sorted version of glob, to ensure determinism and prevent bugs."""
    return sorted(glob.glob(*args, **kwargs))


def parse_val_loss(filename: str) -> float:
    """Parse val_loss from the checkpoint filename."""
    start = filename.index("val_loss=") + len("val_loss=")
    try:
        end = filename.index("-v1.ckpt")
    except ValueError:
        end = filename.index(".ckpt")
    val_loss = float(filename[start:end])
    return val_loss


def get_best_val_checkpoint(
    checkpoint_dir,
) -> Union[Tuple[str, np.float64], Tuple[None, None]]:
    """Find the checkpoint with the best val_loss in the checkpoint directory."""
    checkpoints = sorted_glob(os.path.join(checkpoint_dir, "*val*.ckpt"))
    if len(checkpoints) == 0:
        return None, None
    losses = np.array([parse_val_loss(checkpoint) for checkpoint in checkpoints])
    argmin = np.argmin(losses)
    return checkpoints[argmin], losses[argmin]


def get_checkpoints(
    out_directory: str,
    last_checkpoints_too: bool = False,
) -> Tuple[List[str], List[Dict[str, Union[int, float, str]]]]:
    """Gather checkpoint filenames and attribute dictionaries from output directory."""
    checkpoints = []
    attribute_dicts = []

    checkpoint_dir = os.path.join(out_directory, train.checkpoint_dir)
    val_checkpoint, val_loss = get_best_val_checkpoint(checkpoint_dir)
    if val_checkpoint is not None:
        checkpoints.append(val_checkpoint)
        attribute_dicts.append(
            {"Checkpoint": "Validation", "val_loss": val_loss},
        )
    else:
        last_checkpoints_too = True
    if last_checkpoints_too:
        last_checkpoint = sorted_glob(os.path.join(checkpoint_dir, "last.ckpt"))[-1]
        checkpoints.append(last_checkpoint)
        attribute_dicts.append({"Checkpoint": "Last"})

    return checkpoints, attribute_dicts


def get_parameters(out_directory: str) -> Dict[str, Union[int, float, str, bool]]:
    """Load parameters from the output directory."""
    args_file = os.path.join(out_directory, train.args_filename)
    with open(args_file, "r") as f:
        parameters = json.load(f)

    return parameters


def load_experiment(
    out_directory: str,
    last_checkpoints_too: bool = False,
) -> Tuple[
    List[str],
    List[Dict[str, Union[int, float, str]]],
    Dict[str, Union[int, float, str, bool]],
    Union[step.GCSLToGym, offline_env.OfflineEnv],
]:
    """Load experiment from the output directory.

    Returns paths to model checkpoints, their associated attribute dictionaries, the
    parameters of the experimental run, and the environment.
    """
    checkpoints, attribute_dicts = get_checkpoints(
        out_directory,
        last_checkpoints_too=last_checkpoints_too,
    )
    parameters = get_parameters(out_directory)
    parameters["unconditional_policy"] = parameters.get("unconditional_policy", False)
    parameters["reward_conditioning"] = parameters.get("reward_conditioning", False)
    parameters["cumulative_reward_to_go"] = parameters.get(
        "cumulative_reward_to_go",
        False,
    )
    parameters["seed"] = parameters.get("seed", None)

    set_seed(parameters["seed"])
    env = step.create_env(
        parameters["env_name"],
        parameters["max_episode_steps"],
        parameters["discretize"],
        seed=parameters["seed"],
    )

    return checkpoints, attribute_dicts, parameters, env

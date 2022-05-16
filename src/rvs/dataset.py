"""Builds the data modules used to train the policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
import random
from typing import Dict, Iterator, List, Optional, Tuple, Union

from d4rl import offline_env
import gym
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils import data

from rvs import step, util

max_num_workers = 16


def create_data_module(
    env: gym.Env,
    env_name: str,
    rollout_directory: str,
    batch_size: int = 256,
    val_frac: float = 0.1,
    unconditional_policy: bool = False,
    reward_conditioning: bool = False,
    average_reward_to_go: bool = True,
    seed: Optional[int] = None,
) -> AbstractDataModule:
    """Creates the data module used for training."""
    if unconditional_policy and reward_conditioning:
        raise ValueError("Cannot condition on reward with an unconditional policy.")

    if env_name in step.d4rl_env_names:
        if unconditional_policy:
            data_module = D4RLBCDataModule(
                env,
                batch_size=batch_size,
                val_frac=val_frac,
                seed=seed,
            )
        elif reward_conditioning:
            data_module = D4RLRvSRDataModule(
                env,
                batch_size=batch_size,
                val_frac=val_frac,
                average_reward_to_go=average_reward_to_go,
                seed=seed,
            )
        else:
            data_module = D4RLRvSGDataModule(
                env,
                batch_size=batch_size,
                seed=seed,
                val_frac=val_frac,
            )
    else:
        if unconditional_policy:
            raise NotImplementedError
        else:
            data_module = GCSLDataModule(
                rollout_directory,
                batch_size=batch_size,
                val_frac=val_frac,
                seed=seed,
                num_workers=os.cpu_count(),
            )

    return data_module


def s_g_pair_iter(
    s_obs_vecs: np.ndarray,
    s_ach_goal_vecs: np.ndarray,
    a_vecs: np.ndarary,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Use hindsight to iterate over all states and future achieved goals."""
    for episode in range(len(a_vecs)):
        s_obs_vec = s_obs_vecs[episode]
        s_ach_goal_vec = s_ach_goal_vecs[episode]
        a_vec = a_vecs[episode]
        for i in range(len(a_vec)):
            for j in range(i + 1, len(s_obs_vec)):
                s = s_obs_vec[i]
                a = a_vec[i]
                g = s_ach_goal_vec[j]
                yield s, a, g


def make_s_g_tensor(states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
    """Combine observations and goals into the same tensor."""
    s_tensor = torch.tensor(states)
    g_tensor = torch.tensor(goals)
    s_g_tensor = torch.cat((s_tensor, g_tensor), dim=1)

    return s_g_tensor


def to_tensor_dataset(
    data_vec: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> data.TensorDataset:
    """Convert a list of data into a tensor dataset."""
    states, goals, actions = zip(*data_vec)
    states = np.array(states)
    goals = np.array(goals)

    s_g_tensor = make_s_g_tensor(states, goals)
    a_tensor = torch.tensor(actions)

    assert not s_g_tensor.requires_grad
    assert not a_tensor.requires_grad

    return data.TensorDataset(s_g_tensor, a_tensor)


def load_tensor_dataset(rollout_directory: str) -> data.TensorDataset:
    """Load saved GCSL rollouts and return a tensor dataset."""
    s_obs_vecs = np.load(os.path.join(rollout_directory, step.s_obs_vecs_file))
    s_ach_goal_vecs = np.load(
        os.path.join(rollout_directory, step.s_ach_goal_vecs_file),
    )
    a_vecs = np.load(os.path.join(rollout_directory, step.a_vecs_file))

    data_vec = [
        (s, g, a) for s, a, g in s_g_pair_iter(s_obs_vecs, s_ach_goal_vecs, a_vecs)
    ]
    tensor_dataset = to_tensor_dataset(data_vec)

    return tensor_dataset


def seed_worker(worker_id: int) -> None:
    """Unique random seed for each parallel data worker to prevent duplicate batches."""
    # torch.initial_seed() is the base torch seed plus a unique offset for each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class D4RLIterableDataset(data.IterableDataset):
    """Used for goal-conditioned learning in D4RL."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        dones: np.ndarray,
        epoch_size: int = 2450000,
        index_batch_size: int = 64,
        goal_columns: Optional[Union[Tuple[int], List[int], np.ndarray]] = None,
    ):
        """Initializes the dataset.

        Args:
            observations: The observations for the dataset.
            actions: The actions for the dataset.
            dones: The dones for the dataset.
            epoch_size: For PyTorch Lightning to count epochs.
            index_batch_size: This has no effect on the functionality of the dataset,
                but it is used internally as the batch size to fetch random indices.
            goal_columns: If not None, then only use these columns of the
                observation_space for the goal conditioning.
        """
        super().__init__()

        self.observations = observations
        self.actions = actions
        self.dones = dones
        self.epoch_size = epoch_size
        self.index_batch_size = index_batch_size
        self.goal_columns = goal_columns

    def _sample_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        starts, ends, lengths = util.extract_done_markers(self.dones)

        # Credit to Dibya Ghosh's GCSL codebase for the logic in the following block:
        # https://github.com/dibyaghosh/gcsl/blob/
        # cfae5609cee79e5a2228fb7653451023c41a64cb/gcsl/algo/buffer.py#L78
        trajectory_indices = np.random.choice(len(starts), self.index_batch_size)
        proportional_indices_1 = np.random.rand(self.index_batch_size)
        proportional_indices_2 = np.random.rand(self.index_batch_size)
        time_indices_1 = np.floor(
            proportional_indices_1 * (lengths[trajectory_indices] - 1),
        ).astype(int)
        time_indices_2 = np.floor(
            proportional_indices_2 * lengths[trajectory_indices],
        ).astype(int)

        start_indices = starts[trajectory_indices] + np.minimum(
            time_indices_1,
            time_indices_2,
        )
        goal_indices = starts[trajectory_indices] + np.maximum(
            time_indices_1,
            time_indices_2,
        )

        return start_indices, goal_indices

    def _sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        start_indices, goal_indices = self._sample_indices()

        observation_batch = self.observations[start_indices]
        goal_batch = self.observations[goal_indices]
        if self.goal_columns is not None:
            goal_batch = np.take(goal_batch, self.goal_columns, axis=1)
        action_batch = self.actions[start_indices]

        return observation_batch, goal_batch, action_batch

    def __iter__(self) -> Iterator[Tuple[torch.tensor, torch.tensor]]:
        """Yield each training example."""
        examples_yielded = 0
        while examples_yielded < self.epoch_size:
            (
                observation_batch,
                goal_batch,
                action_batch,
            ) = self._sample_batch()

            observation_tensors = torch.tensor(observation_batch)
            goal_tensors = torch.tensor(goal_batch)
            action_tensors = torch.tensor(action_batch)

            for observation, goal, action in zip(
                observation_tensors,
                goal_tensors,
                action_tensors,
            ):
                yield torch.cat((observation, goal), dim=0), action
                examples_yielded += 1
                if examples_yielded >= self.epoch_size:
                    break

    def __len__(self) -> int:
        """The number of examples in an epoch. Used by the trainer to count epochs."""
        return self.epoch_size


class AbstractDataModule(pl.LightningDataModule, ABC):
    """Abstract class that serves as parent for all DataModules."""

    def __init__(
        self,
        batch_size: int = 256,
        val_frac: float = 0.1,
        num_workers: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """Initialization for the abstract class.

        Args:
            batch_size: How many examples to return per batch.
            val_frac: What fraction of examples to use as a validation set.
            num_workers: How many cpu workers to fetch data. If not specified, takes
                the minimum of os.cpu_count() and max_num_workers (defined at the top of
                this file).
            seed: A seed for the random dataset samples.
        """
        super().__init__()
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.num_workers = num_workers or min(os.cpu_count(), max_num_workers)

        # These should be created in self.setup()
        self.data_train = None
        self.data_val = None

        if seed is None:
            seed = np.random.randint(2**31 - 1)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    @abstractmethod
    def setup(self, *args, **kwargs) -> None:
        """Create the training and validation data."""
        pass

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self) -> data.DataLoader:
        """Make the training dataloader."""
        return data.DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=self.generator,
            worker_init_fn=seed_worker,
        )

    def val_dataloader(self) -> data.DataLoader:
        """Make the validation dataloader."""
        return data.DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=self.generator,
            worker_init_fn=seed_worker,
        )


class GCSLDataModule(AbstractDataModule):
    """The data module used for GCSL envs."""

    def __init__(
        self,
        rollout_directory: str,
        batch_size: int = 32,
        val_frac: float = 0.2,
        num_workers: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """Custom initialization for the GCSL data module."""
        super().__init__(
            batch_size=batch_size,
            val_frac=val_frac,
            num_workers=num_workers,
            seed=seed,
        )
        self.rollout_directory = rollout_directory

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the training and validation data."""
        tensor_dataset = load_tensor_dataset(self.rollout_directory)
        n_val = int(self.val_frac * len(tensor_dataset))
        n_train = len(tensor_dataset) - n_val

        data_train, data_val = data.random_split(tensor_dataset, [n_train, n_val])
        if n_val == 0:
            data_val = None

        if stage == "fit" or stage is None:
            self.data_train, self.data_val = data_train, data_val


def d4rl_trajectory_split(
    dones: np.ndarray,
    val_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Divides the D4RL trajectories into training and validation splits.

    Args:
        dones: Flags indicating the ends of trajectories.
        val_frac: What fraction of the trajectories to put in the validation set.

    Returns:
        Indices indicating which timesteps go in the training set and which go in the
            validation set.
    """
    assert 0 <= val_frac <= 1

    starts, ends, lengths = util.extract_done_markers(dones)
    n_val = int(val_frac * len(starts))
    n_train = len(starts) - n_val

    train_traj_indices = np.arange(n_train)
    val_traj_indices = np.arange(n_val) + n_train

    # avoid biased splits when trajectories are ordered, e.g., in combined datasets
    shuffled = np.arange(n_train + n_val)
    np.random.shuffle(shuffled)
    train_traj_indices = shuffled[train_traj_indices]
    val_traj_indices = shuffled[val_traj_indices]

    train_indices = util.collect_timestep_indices(dones, train_traj_indices).astype(int)
    val_indices = util.collect_timestep_indices(dones, val_traj_indices).astype(int)

    return train_indices, val_indices


def reward_to_go(dataset: Dict[str, np.ndarray], average: bool = True) -> np.ndarray:
    """Compute the reward to go for each timestep.

    The implementation is iterative because when I wrote a vectorized version, np.cumsum
    cauased numerical instability.
    """
    dones = np.logical_or(dataset["terminals"], dataset["timeouts"])
    _, _, lengths = util.extract_done_markers(dones)
    max_episode_steps = np.max(lengths)

    reverse_reward_to_go = np.inf * np.ones_like(dataset["rewards"])
    running_reward = 0
    for i, (reward, done) in enumerate(zip(dataset["rewards"][::-1], dones[::-1])):
        if done:
            running_reward = 0
        running_reward += reward
        reverse_reward_to_go[i] = running_reward
    cum_reward_to_go = reverse_reward_to_go[::-1].copy()

    avg_reward_to_go = np.inf * np.ones_like(cum_reward_to_go)
    elapsed_time = 0
    for i, (cum_reward, done) in enumerate(zip(cum_reward_to_go, dones)):
        avg_reward_to_go[i] = cum_reward / (max_episode_steps - elapsed_time)
        elapsed_time += 1
        if done:
            elapsed_time = 0

    return avg_reward_to_go if average else cum_reward_to_go


class D4RLTensorDatasetDataModule(AbstractDataModule):
    """Abstract class for D4RL datasets that can be stored as a TensorDataset."""

    def __init__(
        self,
        env: offline_env.OfflineEnv,
        batch_size: int,
        val_frac: float = 0.1,
        num_workers: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """Custom initialization that saves the environment."""
        super().__init__(
            batch_size=batch_size,
            val_frac=val_frac,
            num_workers=num_workers,
            seed=seed,
        )
        self.env = env

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the training and validation data."""
        dataset = self.env.get_dataset()
        observation_tensor = self._get_observation_tensor(dataset)
        action_tensor = torch.tensor(dataset["actions"])
        dones = np.logical_or(dataset["terminals"], dataset["timeouts"])

        train_indices, val_indices = d4rl_trajectory_split(dones, self.val_frac)
        train_dataset = data.TensorDataset(
            observation_tensor[train_indices],
            action_tensor[train_indices],
        )
        val_dataset = (
            data.TensorDataset(
                observation_tensor[val_indices],
                action_tensor[val_indices],
            )
            if self.val_frac > 0
            else None
        )

        if stage == "fit" or stage is None:
            self.data_train, self.data_val = train_dataset, val_dataset

    @abstractmethod
    def _get_observation_tensor(self, dataset: Dict[str, np.ndarray]) -> torch.Tensor:
        pass


class D4RLBCDataModule(D4RLTensorDatasetDataModule):
    """Data module for unconditional behavior cloning in D4RL."""

    def _get_observation_tensor(self, dataset: Dict[str, np.ndarray]) -> torch.Tensor:
        return torch.tensor(dataset["observations"])


class D4RLRvSRDataModule(D4RLTensorDatasetDataModule):
    """Data module for RvS-R (reward-conditioned) learning in D4RL."""

    def __init__(
        self,
        env: offline_env.OfflineEnv,
        batch_size: int,
        val_frac: float = 0.1,
        num_workers: Optional[int] = None,
        average_reward_to_go: bool = True,
        seed: Optional[int] = None,
    ):
        """Custom initialization that sets the average_reward_to_go."""
        super().__init__(
            env,
            batch_size,
            val_frac=val_frac,
            num_workers=num_workers,
            seed=seed,
        )
        self.average_reward_to_go = average_reward_to_go

    def _get_observation_tensor(self, dataset: Dict[str, np.ndarray]) -> torch.Tensor:
        return make_s_g_tensor(
            dataset["observations"],
            reward_to_go(dataset, average=self.average_reward_to_go).reshape(-1, 1),
        )


class D4RLRvSGDataModule(AbstractDataModule):
    """Data module for RvS-G (goal-conditioned) learning in D4RL."""

    def __init__(
        self,
        env: offline_env.OfflineEnv,
        batch_size: int = 32,
        val_frac: float = 0.1,
        num_workers: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """Custom initialization.

        Saves the environment and conditions on the (x, y) coordinate of the goal in
        AntMaze.
        """
        super().__init__(
            batch_size=batch_size,
            val_frac=val_frac,
            num_workers=num_workers,
            seed=seed,
        )
        self.env = env
        self.goal_columns = (0, 1) if step.is_antmaze_env(env) else None

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the training and validation data."""
        dataset = self.env.get_dataset()
        observations = dataset["observations"]
        actions = dataset["actions"]
        if step.is_antmaze_env(self.env):
            dones = dataset["timeouts"]
        else:
            dones = np.logical_or(dataset["terminals"], dataset["timeouts"])

        train_indices, val_indices = d4rl_trajectory_split(dones, self.val_frac)

        train_dataset = D4RLIterableDataset(
            observations[train_indices],
            actions[train_indices],
            dones[train_indices],
            goal_columns=self.goal_columns,
        )
        val_dataset = (
            D4RLIterableDataset(
                observations[val_indices],
                actions[val_indices],
                dones[val_indices],
                goal_columns=self.goal_columns,
            )
            if self.val_frac > 0
            else None
        )

        if stage == "fit" or stage is None:
            self.data_train, self.data_val = train_dataset, val_dataset

"""Visualize performance of completed runs."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple, Union

import configargparse
from d4rl import offline_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from wandb.sdk.wandb_run import Run

from rvs import policies, step, util


def save_file_to_wandb(filename: str, wandb_run: Optional[Run] = None) -> None:
    """Save a file to a wandb run."""
    if wandb_run is not None:
        if "lightning_logs" in filename:
            base_path = filename[: filename.find("lightning_logs")]
        else:
            base_path = None
        wandb_run.save(filename, base_path=base_path)


def log_plt_as_image(log_key: str, wandb_run: Optional[Run] = None) -> None:
    """Log the plt as an image in wandb."""
    if wandb_run is not None:
        image = wandb.Image(plt)
        wandb_run.log({log_key: image})


def aggregate_performance(
    performance_vecs: Union[np.ndarray, List[np.ndarray]],
    attribute_dicts: List[Dict[str, Union[int, float, str]]],
    performance_metric: str,
) -> pd.DataFrame:
    """Combine the performance vectors and their attributes into one DataFrame."""
    assert len(performance_vecs) == len(
        attribute_dicts,
    ), "Must have one attribute dict per performance vec"

    df = pd.DataFrame()
    for reward_vec, attribute_dict in zip(performance_vecs, attribute_dicts):
        d = pd.DataFrame(
            {
                performance_metric: reward_vec,
            },
        )
        for key, value in attribute_dict.items():
            d[key] = value
        df = df.append(d, ignore_index=True)

    return df


def get_performance_vec(
    checkpoint_file: str,
    max_episode_steps: int,
    env: Union[step.GCSLToGym, offline_env.OfflineEnv],
    env_name: str,
    device: torch.device,
    hitting_time_samples: int = 2000,
    force_rollouts: bool = False,
    kitchen_subtask: str = "all",
    unconditional_policy: bool = False,
    reward_conditioning: bool = False,
    wandb_run: Optional[Run] = None,
) -> np.ndarray:
    """Load the policy checkpoint and sample its performance."""
    performance_file, _ = os.path.splitext(checkpoint_file)

    try:
        if not force_rollouts:
            performance_vec = np.load(performance_file + ".npy")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        policy = policies.RvS.load_from_checkpoint(
            checkpoint_file,
            map_location=device,
            observation_space=env.observation_space,
            action_space=env.action_space,
            unconditional_policy=unconditional_policy,
            reward_conditioning=reward_conditioning,
            env_name=env_name,
        )

        performance_vec = step.sample_episode_performance(
            policy,
            env,
            env_name,
            max_episode_steps,
            traj_samples=hitting_time_samples,
            kitchen_subtask=kitchen_subtask,
        )

        np.save(performance_file, performance_vec)
        save_file_to_wandb(performance_file, wandb_run=wandb_run)

    return performance_vec


def get_performance_vecs(
    checkpoints: Iterable[str],
    env: Union[step.GCSLToGym, offline_env.OfflineEnv],
    env_name: str,
    device: torch.device,
    max_episode_steps: int,
    hitting_time_samples: int = 2000,
    force_rollouts: bool = False,
    kitchen_subtask: str = "all",
    unconditional_policy: bool = False,
    reward_conditioning: bool = False,
    wandb_run: Optional[Run] = None,
) -> List[np.ndarray]:
    """Load the policy checkpoints and sample their performance."""
    performance_vecs = []

    for checkpoint_file in checkpoints:
        performance_vec = get_performance_vec(
            checkpoint_file,
            max_episode_steps,
            env,
            env_name,
            device,
            hitting_time_samples=hitting_time_samples,
            force_rollouts=force_rollouts,
            kitchen_subtask=kitchen_subtask,
            unconditional_policy=unconditional_policy,
            reward_conditioning=reward_conditioning,
            wandb_run=wandb_run,
        )
        performance_vecs.append(performance_vec)

    return performance_vecs


def get_episode_rewards(
    timestep_rewards: np.ndarray,
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
) -> np.ndarray:
    """Given rewards for each timestep, calculate rewards for each episode."""
    reward_cumsum = np.cumsum(timestep_rewards)
    episode_rewards = (
        reward_cumsum[episode_ends]
        - reward_cumsum[episode_starts]
        + timestep_rewards[episode_starts]
    )
    return episode_rewards


def get_demonstrator_reward_vec(env: offline_env.OfflineEnv) -> np.ndarray:
    """Calculate the demonstrator's reward for each episode."""
    dataset = env.get_dataset()
    dones = np.logical_or(dataset["terminals"], dataset["timeouts"])
    starts, ends, lengths = util.extract_done_markers(dones)

    episode_rewards = get_episode_rewards(dataset["rewards"], starts, ends)
    return episode_rewards


def save_df_with_plot(
    df: pd.DataFrame,
    plt_filename: str,
    wandb_run: Optional[Run] = None,
    wandb_table_key: str = "df",
) -> None:
    """Save DataFrame as .csv and log it as a wandb table."""
    root, _ = os.path.splitext(plt_filename)
    df.to_csv(root + ".csv")
    if wandb_run is not None:
        wandb_table = wandb.Table(dataframe=df)
        wandb_run.log({wandb_table_key: wandb_table})


def plot_average_hit_times(
    hitting_time_vecs: Union[np.ndarray, List[np.ndarray]],
    attribute_dicts: List[Dict[str, Union[int, float, str]]],
    env_name: str,
    out_directory: str,
    x: str = "Checkpoint",
    y: str = "Hitting Time",
    hue: str = "Checkpoint",
    title: Optional[str] = None,
    wandb_run: Optional[Run] = None,
) -> None:
    """Create a barplot of the average hitting times."""
    # log all hitting time vecs
    df = aggregate_performance(hitting_time_vecs, attribute_dicts, y)

    sns.barplot(x=x, y=y, hue=hue, data=df)
    plt.grid()
    if title:
        plt.title(title)
    # plot average hitting time
    plt_filename_root = f"{env_name.lower()}_avg_hit_times"
    plt_filename = os.path.join(out_directory, plt_filename_root + ".png")
    plt.savefig(plt_filename)
    log_plt_as_image(f"{env_name.lower()}_avg_hit_times", wandb_run=wandb_run)
    plt.close()

    save_df_with_plot(
        df,
        plt_filename,
        wandb_run=wandb_run,
        wandb_table_key=plt_filename_root + "_table",
    )


def calculate_cdf_curve(
    hitting_times: np.ndarray,
    max_episode_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the cdf curve of the given hitting times."""
    sorted_times = np.sort(hitting_times)
    times = np.arange(max_episode_steps + 1)
    comparisons = sorted_times[np.newaxis, :] <= times[:, np.newaxis]
    accumulated_probs = comparisons.mean(axis=1)

    return times, accumulated_probs


def plot_cumulative_hit_times(
    hitting_time_vecs: Union[np.ndarray, List[np.ndarray]],
    names: List[Union[str, int, float]],
    max_episode_steps: int,
    out_directory: str,
    plt_filename: str,
    name_key: str = "Legend",
    title: Optional[str] = None,
    wandb_run: Optional[Run] = None,
) -> None:
    """Create a plot of the cumulative hitting times."""
    df = pd.DataFrame()
    for name, hitting_times in zip(names, hitting_time_vecs):
        times, accumulated_probs = calculate_cdf_curve(hitting_times, max_episode_steps)
        if times[-1] != max_episode_steps - 1:
            times = np.concatenate((times, [max_episode_steps - 1]), axis=0)
            accumulated_probs = np.concatenate(
                (accumulated_probs, [accumulated_probs[-1]]),
                axis=0,
            )
        d = pd.DataFrame(
            {
                name_key: name,
                "Hitting Time": times,
                "Cumulative Probability": accumulated_probs,
            },
        )
        df = df.append(d, ignore_index=True)

    sns.lineplot(
        x="Hitting Time",
        y="Cumulative Probability",
        hue=name_key,
        style=name_key,
        data=df,
        palette="viridis",
        n_boot=10000,
    )
    plt.grid()
    if title is not None:
        plt.title(title)
    plt_filename_root, _ = os.path.splitext(plt_filename)
    plt_filename = os.path.join(out_directory, plt_filename)
    plt.savefig(plt_filename)
    log_plt_as_image(plt_filename_root, wandb_run=wandb_run)
    plt.close()

    save_df_with_plot(
        df,
        plt_filename,
        wandb_run=wandb_run,
        wandb_table_key=plt_filename_root + "_table",
    )

    print(f"Visualized hitting times located at {out_directory}")


def visualize_hitting_times(
    hitting_time_vecs: Union[np.ndarray, List[np.ndarray]],
    attribute_dicts: List[Dict[str, Union[int, float, str]]],
    parameters: Dict[str, Union[int, float, str, bool]],
    out_directory: str,
    selection_key: str = "Checkpoint",
    selection_value: str = "Last",
    name_key: str = "Legend",
    title: Optional[str] = None,
    wandb_run: Optional[Run] = None,
) -> None:
    """Plot both the average and cdf of the hitting times."""
    plot_average_hit_times(
        hitting_time_vecs,
        attribute_dicts,
        parameters["env_name"],
        out_directory,
        wandb_run=wandb_run,
    )

    cdf_performance_vecs, cdf_names = zip(
        *[
            (hitting_time_vec, attribute_dict[name_key])
            for hitting_time_vec, attribute_dict in zip(
                hitting_time_vecs,
                attribute_dicts,
            )
            if attribute_dict[selection_key] == selection_value
        ],
    )
    lower_env_name = parameters["env_name"].lower()  # pytype: disable=attribute-error
    plot_cumulative_hit_times(
        cdf_performance_vecs,
        cdf_names,
        parameters["max_episode_steps"],
        out_directory,
        f"{lower_env_name}_cum_hit_times.png",
        name_key=name_key,
        title=title,
        wandb_run=wandb_run,
    )


def visualize_cumulative_reward(
    reward_vecs: Union[np.ndarray, List[np.ndarray]],
    attribute_dicts: List[Dict[str, Union[int, float, str]]],
    parameters: Dict[str, Union[int, float, str, bool]],
    out_directory: str,
    x: str = "Checkpoint",
    y: str = "Return",
    hue: str = "Checkpoint",
    file_tag: str = "Iter",
    title: Optional[str] = None,
    wandb_run: Optional[Run] = None,
) -> None:
    """Create a violin plot of the rewards."""
    df = aggregate_performance(reward_vecs, attribute_dicts, y)

    sns.violinplot(x=x, y=y, hue=hue, data=df)
    plt.grid()
    if title:
        plt.title(title)
    lower_env_name = parameters["env_name"].lower()  # pytype: disable=attribute-error
    lower_file_tag = file_tag.lower()
    plt_filename = f"{lower_env_name}_{lower_file_tag}_reward_violin.png"
    plt_filename = os.path.join(out_directory, plt_filename)
    plt.savefig(plt_filename)
    log_plt_as_image(f"{file_tag.lower()}_reward_violin", wandb_run=wandb_run)
    plt.close()

    save_df_with_plot(
        df,
        plt_filename,
        wandb_run=wandb_run,
        wandb_table_key=f"{file_tag.lower()}_reward_table",
    )


def visualize_performance(
    out_directory: str,
    device: torch.device,
    trajectory_samples: int = 2000,
    force_rollouts: bool = False,
    kitchen_subtask: str = "all",
    wandb_run: Optional[Run] = None,
) -> None:
    """Visualize the performance: hitting times for GCSL, and reward for D4RL."""
    checkpoints, attribute_dicts, parameters, env = util.load_experiment(
        out_directory,
        last_checkpoints_too=True,
    )
    performance_vecs = get_performance_vecs(
        checkpoints,
        env,
        parameters["env_name"],
        device,
        parameters["max_episode_steps"],
        hitting_time_samples=trajectory_samples,
        force_rollouts=force_rollouts,
        kitchen_subtask=kitchen_subtask,
        unconditional_policy=parameters["unconditional_policy"],
        reward_conditioning=parameters["reward_conditioning"],
        wandb_run=wandb_run,  # pytype: disable=wrong-arg-types
    )

    if parameters["env_name"] in step.d4rl_env_names:
        demonstrator_reward_vec = get_demonstrator_reward_vec(env)
        demonstrator_attribute_dict = {"Policy": "Demonstrator"}
        visualize_cumulative_reward(
            performance_vecs + [demonstrator_reward_vec],
            attribute_dicts + [demonstrator_attribute_dict],
            parameters,
            out_directory,
            wandb_run=wandb_run,
        )
    else:
        visualize_hitting_times(
            performance_vecs,
            attribute_dicts,
            parameters,
            out_directory,
            name_key="Checkpoint",
            wandb_run=wandb_run,
        )


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description="Visualize performance of completed runs.",
    )
    parser.add_argument(
        "--run_id",
        required=True,
        type=str,
        help="wandb run id of the experiment to visualize",
    )
    parser.add_argument(
        "--entity",
        required=True,
        type=str,
        help="wandb entity (username) of the experiment to analyze",
    )
    parser.add_argument(
        "--trajectory_samples",
        "--hitting_time_samples",
        default=2000,
        type=int,
        help="the number of trajectory samples used to estimate the hitting time "
        "distribution / reward",
    )
    parser.add_argument(
        "--force_rollouts",
        action="store_true",
        default=False,
        help="simulate environment rollouts to determine GR3 performance even if there "
        "exists a saved performance vector",
    )
    parser.add_argument(
        "--kitchen_subtask",
        default="all",
        type=str,
        help="which subtask to command in D4RL's kitchen environment",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=False,
        help="place networks and data on the GPU",
    )
    parser.add_argument("--which_gpu", default=0, type=int, help="which GPU to use")

    args = parser.parse_args()
    out_directory, wandb_run = util.resolve_out_directory(args.run_id, args.entity)
    device = util.configure_gpu(args.use_gpu, args.which_gpu)
    visualize_performance(
        out_directory,
        device,
        trajectory_samples=args.trajectory_samples,
        force_rollouts=args.force_rollouts,
        kitchen_subtask=args.kitchen_subtask,
        wandb_run=wandb_run,
    )

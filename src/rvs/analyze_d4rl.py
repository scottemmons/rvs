"""Analyze completed D4RL training runs."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Union

import configargparse
from d4rl import offline_env
from d4rl.kitchen import kitchen_envs
from d4rl.locomotion import ant
import numpy as np
import torch
from wandb.sdk.wandb_run import Run

from rvs import policies, step, util, visualize

d4rl_weight_directory = "d4rl_weights"


def run_reward_conditioning(
    out_directory: str,
    parameters: Dict[str, Union[int, float, str, bool]],
    loaded_policies: Iterable[policies.RvS],
    attribute_dicts: List[Dict[str, Union[int, float, str]]],
    env: offline_env.OfflineEnv,
    trajectory_samples: int = 200,
    file_tag: str = "r_target",
    targets: str = "of expert",
    wandb_run: Optional[Run] = None,
) -> None:
    """Evaluate the policies for various reward fractions, and visualize the results."""
    reward_fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    reward_vecs, r_attribute_dicts = [], []
    for policy, attribute_dict in zip(loaded_policies, attribute_dicts):
        reward_vecs += step.eval_reward_conditioning(
            policy,
            env,
            parameters["env_name"],
            reward_fractions,
            trajectory_samples=trajectory_samples,
            targets=targets,
            average_reward_to_go=not parameters["cumulative_reward_to_go"],
        )
        r_attribute_dicts += [
            {"Policy": "RCBC", "Reward Target": reward_fraction, **attribute_dict}
            for reward_fraction in reward_fractions
        ]
    if (
        "antmaze" not in parameters["env_name"]
    ):  # wandb tables can't handle how many terminal flags are in antmaze
        reward_vecs.append(visualize.get_demonstrator_reward_vec(env))
        r_attribute_dicts.append({"Policy": "Demonstrator"})
    visualize.visualize_cumulative_reward(
        reward_vecs,
        r_attribute_dicts,
        parameters,
        out_directory,
        x="Reward Target",
        file_tag=file_tag,
        wandb_run=wandb_run,
    )


def compare_commands_to_demonstrator(
    out_directory: str,
    parameters: Dict[str, Union[int, float, str, bool]],
    loaded_policies: Iterable[policies.RvS],
    attribute_dicts: List[Dict[str, Union[int, float, str]]],
    env: offline_env.OfflineEnv,
    goals: Union[np.ndarray, List[np.ndarray]],
    goal_names: List[Union[str, int, float]],
    file_tag: str = "Iter",
    title: Optional[str] = None,
    trajectory_samples: int = 200,
    dynamic_demonstrator: bool = False,
    wandb_run: Optional[Run] = None,
) -> None:
    """Evaluate the policies and compare their performance to the demonstrations."""
    assert len(goals) == len(goal_names)

    all_reward_vecs = []
    all_attribute_dicts = []
    for loaded_policy, attribute_dict in zip(loaded_policies, attribute_dicts):
        all_reward_vecs += step.evaluate_goals(
            loaded_policy,
            env,
            goals,
            trajectory_samples=trajectory_samples,
        )
        all_attribute_dicts += [
            {**attribute_dict, "Goal": goal_name} for goal_name in goal_names
        ]

        if dynamic_demonstrator:
            all_reward_vecs.append(
                step.sample_episode_performance(
                    loaded_policy,
                    env,
                    parameters["env_name"],
                    parameters["max_episode_steps"],
                    traj_samples=trajectory_samples,
                    kitchen_subtask="dynamic",
                ),
            )
            all_attribute_dicts.append({**attribute_dict, "Goal": "dynamic"})

    if not dynamic_demonstrator:
        all_reward_vecs.append(visualize.get_demonstrator_reward_vec(env))
        all_attribute_dicts.append({"Policy": "Demonstrator"})

    visualize.visualize_cumulative_reward(
        all_reward_vecs,
        all_attribute_dicts,
        parameters,
        out_directory,
        x="Goal",
        file_tag=file_tag,
        title=title,
        wandb_run=wandb_run,
    )


def command_kitchen_subtasks(
    out_directory: str,
    parameters: Dict[str, Union[int, float, str, bool]],
    loaded_policies: Iterable[policies.RvS],
    attribute_dicts: List[Dict[str, Union[int, float, str]]],
    env: kitchen_envs.KitchenBase,
    trajectory_samples: int = 200,
    wandb_run: Optional[Run] = None,
) -> None:
    """Evaluate the reward when choosing different kitchen subtasks as goals."""
    valid_subtasks = step.get_valid_kitchen_subtasks(env)
    goals = [step.get_kitchen_goal(env, subtask)[0] for subtask in valid_subtasks]
    compare_commands_to_demonstrator(
        out_directory,
        parameters,
        loaded_policies,
        attribute_dicts,
        env,
        goals,
        valid_subtasks,
        file_tag="kitchen_subtasks",
        title=f"Commanding Subtasks in {parameters['env_name']}",
        trajectory_samples=trajectory_samples,
        dynamic_demonstrator=True,
        wandb_run=wandb_run,
    )


def analyze_antmaze(
    out_directory: str,
    parameters: Dict[str, Union[int, float, str, bool]],
    loaded_policies: Iterable[policies.RvS],
    attribute_dicts: List[Dict[str, Union[int, float, str]]],
    env: ant.AntMazeEnv,
    trajectory_samples: int = 200,
    wandb_run: Optional[Run] = None,
) -> None:
    """Analyze the performance of an AntMaze run."""
    reward_vecs = [
        step.eval_d4rl_antmaze(policy, env, trajectory_samples=trajectory_samples)
        for policy in loaded_policies
    ]
    visualize.visualize_cumulative_reward(
        reward_vecs,
        attribute_dicts,
        parameters,
        out_directory,
        wandb_run=wandb_run,
    )


def use_elite_goals(
    out_directory: str,
    parameters: Dict[str, Union[int, float, str, bool]],
    loaded_policies: List[policies.RvS],
    attribute_dicts: List[Dict[str, Union[int, float, str]]],
    env: offline_env.OfflineEnv,
    trajectory_samples: int = 200,
    wandb_run: Optional[Run] = None,
) -> None:
    """Find, evaluate, and visualize the best-performing length goals."""
    goals = step.find_elite_goals(
        loaded_policies[0],
        env,
        trajectory_samples=trajectory_samples,
    )
    goal_names = [f"Elite {i}" for i in range(len(goals))]
    compare_commands_to_demonstrator(
        out_directory,
        parameters,
        loaded_policies,
        attribute_dicts,
        env,
        goals,
        goal_names,
        file_tag="elite_goal",
        title=f"Expert Goals in {parameters['env_name']}",
        trajectory_samples=trajectory_samples,
        wandb_run=wandb_run,
    )


def vary_commanded_goal(
    out_directory: str,
    parameters: Dict[str, Union[int, float, str, bool]],
    loaded_policies: Iterable[policies.RvS],
    attribute_dicts: List[Dict[str, Union[int, float, str]]],
    env: offline_env.OfflineEnv,
    trajectory_samples: int = 200,
    elite_property: str = "Length",
    elite_traj_fraction: float = 0.05,
    elite_step_fraction: float = 0.05,
    wandb_run: Optional[Run] = None,
) -> None:
    """Extract, evaluate, and visualize the best goals according to elite property."""
    goals, _ = step.sample_elite_steps(
        env.get_dataset(),
        samples=6,
        elite_property=elite_property.lower(),
        elite_traj_fraction=elite_traj_fraction,
        elite_step_fraction=elite_step_fraction,
    )
    goal_names = [f"{elite_property} {i}" for i in range(len(goals))]
    compare_commands_to_demonstrator(
        out_directory,
        parameters,
        loaded_policies,
        attribute_dicts,
        env,
        goals,
        goal_names,
        file_tag=f"{elite_property.lower()}_goal",
        title=f"{elite_property} Goals in {parameters['env_name']}",
        trajectory_samples=trajectory_samples,
        wandb_run=wandb_run,
    )


def analyze_performance(
    out_directory: str,
    device: torch.device,
    trajectory_samples: int = 200,
    analysis: str = "all",
    wandb_run: Optional[Run] = None,
    last_checkpoints_too: bool = True,
) -> None:
    """Main method that calls the appropriate helper method to run the analysis."""
    checkpoints, attribute_dicts, parameters, env = util.load_experiment(
        out_directory,
        last_checkpoints_too=last_checkpoints_too,
    )

    loaded_policies = [
        policies.RvS.load_from_checkpoint(
            checkpoint,
            map_location=device,
            observation_space=env.observation_space,
            action_space=env.action_space,
            unconditional_policy=parameters["unconditional_policy"],
            reward_conditioning=parameters["reward_conditioning"],
            env_name=parameters["env_name"],
        )
        for checkpoint in checkpoints
    ]

    # with reward conditioning, just analyze reward with varying reward targets
    if parameters["reward_conditioning"]:
        run_reward_conditioning(
            out_directory,
            parameters,
            loaded_policies,
            attribute_dicts,
            env,
            trajectory_samples=trajectory_samples,
            wandb_run=wandb_run,
        )
        return
    # AntMaze requires its own analysis logic
    if step.is_antmaze_env(env):
        analyze_antmaze(
            out_directory,
            parameters,
            loaded_policies,
            attribute_dicts,
            env,
            trajectory_samples=trajectory_samples,
            wandb_run=wandb_run,
        )
        return
    # command the various kitchen subtasks
    if (analysis == "kitchen_subtasks" or analysis == "all") and step.is_kitchen_env(
        env,
    ):
        command_kitchen_subtasks(
            out_directory,
            parameters,
            loaded_policies,
            attribute_dicts,
            env,
            trajectory_samples=trajectory_samples,
            wandb_run=wandb_run,
        )
    # does conditioning on the best commanded goals lead to better performance?
    if analysis == "elite_goals" or analysis == "all":
        use_elite_goals(
            out_directory,
            parameters,
            loaded_policies,
            attribute_dicts,
            env,
            trajectory_samples=trajectory_samples,
            wandb_run=wandb_run,
        )
    # create performance violins for commanded goals from long trajectories
    if analysis == "length_goals" or analysis == "all":
        vary_commanded_goal(
            out_directory,
            parameters,
            loaded_policies,
            attribute_dicts,
            env,
            trajectory_samples=trajectory_samples,
            elite_property="Length",
            wandb_run=wandb_run,
        )
    # create performance violins for commanded goals from high-reward trajectories
    if analysis == "reward_goals" or analysis == "all":
        vary_commanded_goal(
            out_directory,
            parameters,
            loaded_policies,
            attribute_dicts,
            env,
            trajectory_samples=trajectory_samples,
            elite_property="Reward",
            wandb_run=wandb_run,
        )


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description="analyze properties of RvS performance in D4RL envs",
    )
    parser.add_argument(
        "--run_id",
        required=True,
        type=str,
        help="wandb run id of the experiment to analyze",
    )
    parser.add_argument(
        "--entity",
        required=True,
        type=str,
        help="wandb entity (username) of the experiment to analyze",
    )
    parser.add_argument(
        "--trajectory_samples",
        default=200,
        type=int,
        help="the number of trajectory samples used to estimate the reward",
    )
    parser.add_argument(
        "--val_checkpoint_only",
        action="store_true",
        default=False,
        help="only analyze the validation checkpoint (but if it doesn't exist, still "
        "analyze the last checkpoint)",
    )
    parser.add_argument(
        "--analysis",
        default="all",
        type=str,
        choices=[
            "input_interpolation",
            "weight_histograms",
            "kitchen_subtasks",
            "elite_goals",
            "length_goals",
            "reward_goals",
            "all",
        ],
        help="which analysis to run",
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
    analyze_performance(
        out_directory,
        device,
        trajectory_samples=args.trajectory_samples,
        analysis=args.analysis,
        wandb_run=wandb_run,
        last_checkpoints_too=not args.val_checkpoint_only,
    )

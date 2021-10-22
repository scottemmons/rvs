"""Methods for interacting with the environments."""

# TODO(scottemmons): replace typing imports with __future__ annotations
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import d4rl  # noqa F401 Import needed to register d4rl envs
from d4rl import infos, offline_env
from d4rl.kitchen import kitchen_envs
from d4rl.kitchen.adept_envs.franka import kitchen_multitask_v0
from d4rl.locomotion import ant
from gcsl import envs
from gcsl.algo import variants
from gcsl.envs import goal_env
import gym
import numpy as np
import tqdm

from rvs import dataset, policies, util, visualize

gym_goal_envs = [
    "FetchPickAndPlace-v1",
    "FetchPush-v1",
    "FetchReach-v1",
    "FetchSlide-v1",
    "HandManipulateBlock-v0",
    "HandManipulateEgg-v0",
    "HandManipulatePen-v0",
    "HandReach-v0",
]

d4rl_agents = ["hopper", "halfcheetah", "ant", "walker2d"]
d4rl_datasets = [
    "random",
    "medium",
    "expert",
    "medium-expert",
    "medium-replay",
    "full-replay",
]
d4rl_versions = ["v0", "v1", "v2"]
d4rl_gym = [
    f"{agent}-{dataset}-{version}"
    for agent in d4rl_agents
    for dataset in d4rl_datasets
    for version in d4rl_versions
]
d4rl_antmaze_v0 = [
    "antmaze-umaze-v0",
    "antmaze-umaze-diverse-v0",
    "antmaze-medium-diverse-v0",
    "antmaze-medium-play-v0",
    "antmaze-large-diverse-v0",
    "antmaze-large-play-v0",
]
d4rl_antmaze_v1 = [
    "antmaze-umaze-v1",
    "antmaze-umaze-diverse-v1",
    "antmaze-medium-diverse-v1",
    "antmaze-medium-play-v1",
    "antmaze-large-diverse-v1",
    "antmaze-large-play-v1",
]
d4rl_antmaze = d4rl_antmaze_v0 + d4rl_antmaze_v1
d4rl_franka = ["kitchen-complete-v0", "kitchen-partial-v0", "kitchen-mixed-v0"]
d4rl_env_names = d4rl_gym + d4rl_antmaze + d4rl_franka

s_obs_vecs_file = "s_obs_vecs.npy"
s_ach_goal_vecs_file = "s_ach_goal_vecs.npy"
a_vecs_file = "a_vecs.npy"
base_actions_file = "base_actions.npy"


def save_rollouts(
    rollout_dir: str,
    s_obs_vecs: Union[np.ndarray, List[np.ndarray]],
    s_ach_goal_vecs: Union[np.ndarray, List[np.ndarray]],
    a_vecs: Union[np.ndarray, List[np.ndarray]],
    base_actions: Optional[np.ndarray] = None,
) -> None:
    """Save environment rollouts to the rollout directory."""
    obs_file = os.path.join(rollout_dir, s_obs_vecs_file)
    ach_goal_file = os.path.join(rollout_dir, s_ach_goal_vecs_file)
    act_file = os.path.join(rollout_dir, a_vecs_file)
    base_acts_file = os.path.join(rollout_dir, base_actions_file)

    os.makedirs(rollout_dir, exist_ok=True)
    np.save(obs_file, s_obs_vecs)
    np.save(ach_goal_file, s_ach_goal_vecs)
    np.save(act_file, a_vecs)
    if base_actions is not None:
        np.save(base_acts_file, base_actions)


def load_rollouts(rollout_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load environment rollouts from the rollout directory."""
    obs_file = os.path.join(rollout_dir, s_obs_vecs_file)
    ach_goal_file = os.path.join(rollout_dir, s_ach_goal_vecs_file)
    act_file = os.path.join(rollout_dir, a_vecs_file)

    s_obs_vecs = np.load(obs_file)
    s_ach_goal_vecs = np.load(ach_goal_file)
    a_vecs = np.load(act_file)

    return s_obs_vecs, s_ach_goal_vecs, a_vecs


def get_total_steps(rollout_dir: str) -> int:
    """Calculate the total number of environment steps (trajs * steps / traj)."""
    s_obs_vecs, s_ach_goal_vecs, a_vecs = load_rollouts(rollout_dir)
    assert s_obs_vecs.shape[0] == s_ach_goal_vecs.shape[0] == a_vecs.shape[0]
    assert s_obs_vecs.shape[1] == s_ach_goal_vecs.shape[1] == a_vecs.shape[1]

    total_steps = s_obs_vecs.shape[0] * s_obs_vecs.shape[1]
    return total_steps


def generate_random_rollouts(
    env: GCSLToGym,
    rollout_dir: str,
    total_steps: int,
    max_episode_steps: int,
    use_base_actions: bool = False,
) -> None:
    """Collect random rollouts from the environment.

    Stores observations, actions, and achieved goals.
    """
    try:
        base_actions = env.base_actions
    except AttributeError:
        base_actions = None

    if use_base_actions and base_actions is None:
        raise ValueError("use_base_actions == True but env.base_actions doesn't exist")

    s_obs_vecs = []
    s_ach_goal_vecs = []
    a_vecs = []

    episodes = int(np.ceil(total_steps / max_episode_steps))
    for _ in tqdm.trange(episodes, desc="Generating env episodes"):
        a_vec = []
        s = env.reset()
        s_obs_vec = [s["observation"]]
        s_ach_goal_vec = [s["achieved_goal"]]
        for _ in range(max_episode_steps):
            a = env.action_space.sample()
            s, _, _, _ = env.step(a)
            a_vec.append(a)
            s_obs_vec.append(s["observation"])
            s_ach_goal_vec.append(s["achieved_goal"])
        s_obs_vec = np.array(s_obs_vec)
        s_ach_goal_vec = np.array(s_ach_goal_vec)
        a_vec = np.array(a_vec)

        s_obs_vecs.append(s_obs_vec)
        s_ach_goal_vecs.append(s_ach_goal_vec)
        if use_base_actions:
            a_vecs.append(base_actions[a_vec])
        else:
            a_vecs.append(a_vec)

    save_rollouts(
        rollout_dir,
        s_obs_vecs,
        s_ach_goal_vecs,
        a_vecs,
        base_actions=base_actions,
    )


def is_d4rl_env_helper(env: gym.Env, base_class: Type) -> bool:
    """Helper function that determines if `env` is instance of `base_class`."""
    if isinstance(env, base_class):
        return True
    elif hasattr(env, "env"):
        if isinstance(env.env, base_class):
            return True
        elif hasattr(env.env, "env"):
            if isinstance(env.env.env, base_class):
                return True
        elif hasattr(env.env, "_wrapped_env"):
            if isinstance(env.env._wrapped_env, base_class):
                return True
    return False


def is_kitchen_env(env: gym.Env) -> bool:
    """Determine if env is a D4RL Franka kichen env."""
    return is_d4rl_env_helper(env, kitchen_multitask_v0.KitchenTaskRelaxV1)


def is_antmaze_env(env: gym.Env) -> bool:
    """Determine if env is a D4RL AntMaze env."""
    return is_d4rl_env_helper(env, ant.AntMazeEnv)


def render_env(env: gym.Env, mode="human") -> Union[np.ndarray, None]:
    """Helper function that provides special case for rendering D4RL kitchen envs."""
    if is_kitchen_env(env):
        return kitchen_multitask_v0.KitchenTaskRelaxV1.render(env, mode=mode)
    else:
        return env.render(mode=mode)


def get_action_from_policy(
    policy: Union[policies.RvS, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    obs: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    """Helper function to get action from multiple types of policies."""
    try:
        action = policy.get_action(obs, goal)  # pytype: disable=attribute-error
    except AttributeError:
        action = policy(obs, goal)

    return action


def rollout_and_render(
    policy: Union[policies.RvS, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    env: gym.Env,
    max_episode_steps: int,
    fixed_goal: Optional[np.ndarray] = None,
    dynamic_kitchen_goal: bool = False,
) -> List[np.ndarray]:
    """Roll the policy out in the environment and render every step."""
    frames = []
    if not max_episode_steps:
        max_episode_steps = sys.maxsize

    obs_dict = env.reset()
    frames.append(render_env(env, mode="rgb_array"))
    for _ in range(max_episode_steps):
        if dynamic_kitchen_goal:
            obs = obs_dict
            goal = get_dynamic_kitchen_goal(env, obs)
        elif fixed_goal is not None:
            obs, goal = obs_dict, fixed_goal
        else:
            obs, goal = obs_dict["observation"], obs_dict["desired_goal"]
        a = get_action_from_policy(policy, obs, goal)
        obs_dict, _, done, info = env.step(a)
        frames.append(render_env(env, mode="rgb_array"))

        if done or info.get("is_success", False):
            break

    return frames  # pytype: disable=bad-return-type


def get_valid_kitchen_subtasks(env: kitchen_envs.KitchenBase) -> List[str]:
    """Create list of valid subtasks to command in a D4RL Kitchen env."""
    valid_subtasks = ["all"] + ["random"] + [task for task in env.TASK_ELEMENTS]
    return valid_subtasks


def get_kitchen_goal(
    env: kitchen_envs.KitchenBase,
    subtask: str = "microwave",
    render_goal_frame: bool = False,
) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """Get the goal state and rendered goal frame for the kitchen subtask."""
    valid_subtasks = get_valid_kitchen_subtasks(env)
    if subtask not in valid_subtasks:
        raise ValueError(
            f"Received subtask == {subtask} which is not among the valid choices: "
            f"{valid_subtasks}",
        )

    obs_1 = env.reset()
    obs_2 = env.reset()
    diffs = obs_1 - obs_2
    (zero_indices,) = np.where(diffs == 0)
    assert np.all(zero_indices == np.arange(30, 60))
    fixed = obs_1[30:]

    if subtask == "random":
        goal, goal_frame = get_random_kitchen_goal(
            env,
            render_goal_frame=render_goal_frame,
        )
    else:
        goal = obs_1[:30]
        goal_frame = None
        subtask_collection = env.TASK_ELEMENTS if subtask == "all" else [subtask]
        for task in subtask_collection:
            subtask_indices = kitchen_envs.OBS_ELEMENT_INDICES[task]
            subtask_goals = kitchen_envs.OBS_ELEMENT_GOALS[task]
            goal[subtask_indices] = subtask_goals

    return np.concatenate((goal, fixed), axis=0), goal_frame


def get_random_kitchen_goal(
    env: kitchen_envs.KitchenBase,
    random_horizon: int = 50,
    render_goal_frame: bool = False,
) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """Take random actions and return the achieved state as a goal."""
    observation = env.reset()
    for _ in range(random_horizon):
        random_action = env.action_space.sample()
        observation, _, done, _ = env.step(random_action)
        if done:
            break

    goal = observation[:30]
    goal_frame = render_env(env, mode="rgb_array") if render_goal_frame else None

    return goal, goal_frame


def get_dynamic_kitchen_goal(
    env: kitchen_envs.KitchenBase,
    obs: np.ndarray,
) -> np.ndarray:
    """Get the goal for the next subtask that needs to be completed in the kitchen."""
    goal = np.copy(obs[:30])
    fixed = obs[30:]
    for subtask in env.TASK_ELEMENTS:
        if subtask in env.tasks_to_complete:
            subtask_indices = kitchen_envs.OBS_ELEMENT_INDICES[subtask]
            subtask_goals = kitchen_envs.OBS_ELEMENT_GOALS[subtask]
            goal[subtask_indices] = subtask_goals
            break

    return np.concatenate((goal, fixed), axis=0)


def sample_elite_steps(
    dataset: Dict[str, np.ndarray],
    elite_property: str = "length",
    elite_traj_fraction: float = 0.2,
    elite_step_fraction: float = 0.2,
    samples: int = 200,
    reverse: bool = False,
) -> Tuple[np.ndarray, np.ndarary]:
    """Choose steps from the demonstrations based on a property of the trajectories.

    Args:
        dataset: The dataset of trajectories to use.
        elite_property: Which property of trajectories to use to select the best
            trajectories.
        elite_traj_fraction: If this value is, e.g., 0.2, then sample from the top 20%
            of trajectories.
        elite_step_fraction: If this value is, e.g., 0.2, then sample from the last 20%
            of timesteps.
        samples: How many total steps to return.
        reverse: If true, sample from the worst trajectories (rather than the best).

    Returns:
        A sampled array of observations and a corresponding array of actions.

    Raises:
        ValueError: If an invalid elite property is given.
    """
    starts, ends, lengths = util.extract_traj_markers(dataset)

    if elite_property == "length":
        sorted_indices = np.argsort(lengths)
    elif elite_property == "reward":
        rewards = visualize.get_episode_rewards(dataset["rewards"], starts, ends)
        sorted_indices = np.argsort(rewards)
    else:
        raise ValueError
    if reverse:
        sorted_indices = sorted_indices[::-1]

    num_elites = np.ceil(elite_traj_fraction * len(lengths)).astype(
        int,
    )  # ceil because array indexing is exclusive
    elite_indices = sorted_indices[:-num_elites:-1]
    elite_index = np.random.choice(elite_indices, size=samples)

    elite_start = starts[elite_index]
    elite_proportional_time = (
        1 - elite_step_fraction + np.random.rand(samples) * elite_step_fraction
    )
    elite_relative_time = np.floor(
        elite_proportional_time * lengths[elite_index],
    ).astype(int)
    elite_id = elite_start + elite_relative_time

    return dataset["observations"][elite_id], dataset["actions"][elite_id]


def sample_cumulative_reward(
    policy: Union[policies.RvS, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    env: offline_env.OfflineEnv,
    goals: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    trajectory_samples: int = 2000,
    return_goals: bool = False,
    dynamic_kitchen_goal: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Union[np.ndarray, List[np.ndarray], None]]]:
    """Samples cumulative reward from a D4RL environment.

    Handles the policy conditioning and supports dynamically conditioning on the next
    remaining task in Kitchen.
    """
    if goals is None and not dynamic_kitchen_goal:
        goals, _ = sample_elite_steps(env.get_dataset(), samples=trajectory_samples)
    if goals is not None:
        assert (
            len(goals) == trajectory_samples
        ), "Must have exactly one goal for each trajectory sample"

    total_reward_vec = []
    for i in tqdm.trange(trajectory_samples, desc="Sampling trajectory rewards"):
        total_reward = 0
        observation = env.reset()
        goal = (
            goals[i]
            if not dynamic_kitchen_goal
            else get_dynamic_kitchen_goal(env, observation)
        )
        done = False
        while not done:
            action = get_action_from_policy(policy, observation, goal)
            observation, reward, done, _ = env.step(action)
            if dynamic_kitchen_goal:
                goal = get_dynamic_kitchen_goal(env, observation)
            total_reward += reward
        total_reward_vec.append(total_reward)

    total_rewards = np.array(total_reward_vec)
    if return_goals:
        return total_rewards, goals
    else:
        return total_rewards


def sample_with_reward_conditioning(
    policy: Union[policies.RvS, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    env: gym.Env,
    reward_target: Union[int, float],
    trajectory_samples: int = 200,
) -> np.ndarray:
    """Evaluate cumulative reward with RvS-R (reward-conditioned) learning."""
    total_reward_vec = []
    for _ in tqdm.trange(trajectory_samples, desc="Sampling trajectory rewards"):
        total_reward = 0
        observation = env.reset()
        done = False
        while not done:
            goal = np.array([reward_target - total_reward])
            action = get_action_from_policy(policy, observation, goal)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
        total_reward_vec.append(total_reward)

    total_rewards = np.array(total_reward_vec)
    return total_rewards


def find_elite_goals(
    policy: Union[policies.RvS, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    env: offline_env.OfflineEnv,
    trajectory_samples: int = 200,
    num_elites: int = 6,
) -> np.ndarray:
    """Random search over length goals to find those with highest return."""
    rewards, goals = sample_cumulative_reward(
        policy,
        env,
        trajectory_samples=max(trajectory_samples, num_elites),
        return_goals=True,
    )
    elite_indices = np.argsort(rewards)[
        : -num_elites - 1 : -1
    ]  # indices of num_elites largest rewards
    elite_goals = goals[elite_indices]

    return elite_goals


def evaluate_goals(
    policy: Union[policies.RvS, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    env: offline_env.OfflineEnv,
    goals: Iterable[np.ndarray],
    trajectory_samples: int = 200,
) -> List[np.ndarray]:
    """Evaluate the cumulative return for each goal."""
    reward_vecs = []
    for goal in goals:
        duplicated_goal = [goal] * trajectory_samples
        reward_vec = sample_cumulative_reward(
            policy,
            env,
            goals=duplicated_goal,
            trajectory_samples=trajectory_samples,
        )
        reward_vecs.append(reward_vec)

    return reward_vecs  # pytype: disable=bad-return-type


def sample_hitting_times(
    policy: Union[policies.RvS, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    env: GCSLToGym,
    max_episode_steps: int,
    hitting_time_samples: int = 2000,
) -> np.ndarray:
    """Monte Carlo estimation of the hitting time distribution."""
    hitting_time_vec = []
    for _ in tqdm.trange(hitting_time_samples, desc="Sampling hitting times"):
        hitting_time = 0
        obs_dict = env.reset()
        # As env is wrapped in gym.wrappers.TimeLimit, may not need `max_episode_steps`
        for _ in range(max_episode_steps):
            hitting_time += 1
            obs, goal = obs_dict["observation"], obs_dict["desired_goal"]
            a = get_action_from_policy(policy, obs, goal)
            obs_dict, _, done, info = env.step(a)
            if done or info.get("is_success", False):
                break
        hitting_time_vec.append(hitting_time)
    hitting_times = np.array(hitting_time_vec)
    return hitting_times


def sample_episode_performance(
    policy,
    env: Union[GCSLToGym, offline_env.OfflineEnv],
    env_name: str,
    max_episode_steps: int,
    traj_samples: int = 2000,
    kitchen_subtask: str = "all",
) -> np.ndarray:
    """Helper function to sample episode performance correctly, depending on the env."""
    if env_name[:7] == "kitchen":
        if kitchen_subtask == "dynamic":
            return sample_cumulative_reward(
                policy,
                env,
                trajectory_samples=traj_samples,
                dynamic_kitchen_goal=True,  # pytype: disable=bad-return-type
            )
        else:
            goal, _ = get_kitchen_goal(env, subtask=kitchen_subtask)
            goals = np.repeat(goal[np.newaxis], traj_samples, axis=0)
            return sample_cumulative_reward(
                policy,
                env,
                goals=goals,
                trajectory_samples=traj_samples,  # pytype: disable=bad-return-type
            )
    elif env_name in d4rl_env_names:
        return sample_cumulative_reward(
            policy,
            env,
            trajectory_samples=traj_samples,  # pytype: disable=bad-return-type
        )
    else:
        return sample_hitting_times(
            policy,
            env,
            max_episode_steps,
            hitting_time_samples=traj_samples,
        )


def get_reward_targets(
    env: Union[offline_env.OfflineEnv, gym.wrappers.TimeLimit],
    env_name: str,
    reward_fractions: List[float],
    targets: str = "of expert",
    average_reward_to_go: bool = True,
) -> List[float]:
    """Translate reward fractions into absolute reward targets.

    Args:
        env: The env under consideration.
        env_name: The name of the env.
        reward_fractions: Which reward fractions to convert into cumulative reward.
        targets: Either 'of demos' or 'of expert', indicating w.r.t. what the reward
            fractions are defined.
        average_reward_to_go: If True, use average reward per timestep. Else, use
            cumulative reward for the whole trajectory.

    Returns:
        A list of reward targets that correspond to the given reward fractions.

    Raises:
        ValueError: If an invalid option for targets is specified.
    """
    if targets == "of demos":
        reward_to_go = dataset.reward_to_go(
            env.get_dataset(),
            average=average_reward_to_go,
        )
        reward_min = np.min(reward_to_go)
        reward_max = np.max(reward_to_go)
    elif targets == "of expert":
        if "antmaze" in env_name:
            reward_min = 0
            reward_max = 1
        else:
            reward_min = infos.REF_MIN_SCORE[env_name]
            reward_max = infos.REF_MAX_SCORE[env_name]
        if average_reward_to_go:
            reward_min /= env._max_episode_steps
            reward_max /= env._max_episode_steps
    else:
        raise ValueError("targets must be 'of demos' or 'of expert'")

    reward_targets = [
        reward_min + (reward_max - reward_min) * frac for frac in reward_fractions
    ]
    return reward_targets


def eval_reward_conditioning(
    policy: Union[policies.RvS, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    env: offline_env.OfflineEnv,
    env_name: str,
    reward_fractions: List[float],
    trajectory_samples: int = 200,
    average_reward_to_go: bool = True,
    targets: str = "of expert",
) -> List[np.ndarray]:
    """Evaluate RvS-R (reward-conditioned) learning for each reward fraction."""
    if env_name not in d4rl_env_names:
        raise NotImplementedError

    reward_targets = get_reward_targets(
        env,
        env_name,
        reward_fractions,
        targets=targets,
        average_reward_to_go=average_reward_to_go,
    )

    if average_reward_to_go:
        reward_targets = [np.array([reward_target]) for reward_target in reward_targets]
        reward_vecs = evaluate_goals(
            policy,
            env,
            reward_targets,
            trajectory_samples=trajectory_samples,
        )
    else:
        reward_vecs = []
        for reward_target in reward_targets:
            reward_vec = sample_with_reward_conditioning(
                policy,
                env,
                reward_target,
                trajectory_samples=trajectory_samples,
            )
            reward_vecs.append(reward_vec)

    return reward_vecs


def eval_d4rl_antmaze(
    policy: Union[policies.RvS, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    env: ant.AntMazeEnv,
    trajectory_samples: int = 200,
) -> np.ndarray:
    """Evaluate cumulative reward in AntMaze."""
    assert env.reward_type == "sparse"

    total_reward_vec = []
    for _ in tqdm.trange(trajectory_samples, desc="Sampling trajectory rewards"):
        env.set_target()  # randomly choose new goal
        total_reward = 0
        observation = env.reset()
        goal = np.array(env.target_goal)
        done = False
        while not done:
            assert np.all(np.isclose(observation[:2], env.get_xy()))
            action = get_action_from_policy(policy, observation, goal)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
        total_reward_vec.append(total_reward)

    total_rewards = np.array(total_reward_vec)
    return total_rewards


def random_hitting_times(
    env: GCSLToGym,
    max_episode_steps: int,
    hitting_time_samples: int = 2000,
) -> np.ndarray:
    """For reference, calculate the hitting time of a random policy."""
    random_hitting_time_vec = sample_hitting_times(
        lambda obs, goal: env.action_space.sample(),
        env,
        max_episode_steps,
        hitting_time_samples=hitting_time_samples,
    )
    random_hitting_time = random_hitting_time_vec.mean()
    print(f"Random hitting time: {random_hitting_time:.3f}")

    return random_hitting_time_vec


def seed_env(env: gym.Env, seed: int) -> None:
    """Set the random seed of the environment."""
    if seed is None:
        seed = np.random.randint(2 ** 31 - 1)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def create_env(
    env_name: str,
    max_episode_steps: int,
    discretize: bool,
    seed: Optional[int] = None,
) -> Union[GCSLToGym, offline_env.OfflineEnv]:
    """Helper function to create an environment.

    Args:
        env_name: The name of the environment.
        max_episode_steps: The number of episode steps before the time limit runs out.
        discretize: If True, discretize the environment's action space.
        seed: A random seed for the environment.

    Returns:
        The created environment.

    Raises:
        ValueError: If the environment name is not in GCSL or D4RL.
    """
    if env_name in envs.env_names:
        env = create_gcsl_env(env_name, max_episode_steps, discretize)
    elif env_name in d4rl_env_names:
        env = gym.make(env_name)
    else:
        raise ValueError("Please provide a GCSL or D4RL env name.")

    seed_env(env, seed)
    return env


def create_gcsl_env(
    env_name: str,
    max_episode_steps: int,
    discretize: bool,
) -> GCSLToGym:
    """Create a GCSL environment."""
    env = envs.create_env(env_name)
    env_params = envs.get_env_params(env_name)
    print(env_params)

    if discretize:
        env = variants.discretize_environment(env, env_params)

    env = GCSLToGym(env, goal_threshold=env_params["goal_threshold"])
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    print(env)
    print(env.observation_space)
    print(env.action_space)

    return env


class GCSLToGym(gym.ObservationWrapper):
    """Turn a gcsl.envs.goal_env.GoalEnv into a gym.GoalEnv.

    Note that this wrapper varies slightly from the intended use of the gym.GoalEnv
    interface. In this wrapper, the achieved_goal and the desired_goal are the
    _observations_ of the goal, whereas the gym.GoalEnv wrapper intends for them to
    be the _states_ of the goal.
    """

    def __init__(self, env: goal_env.GoalEnv, goal_threshold: float = 0.05):
        """Wrap a GCSL env into a gym.GoalEnv.

        Args:
            env: The GCSL env to wrap.
            goal_threshold: The distance required to reach the goal.
        """
        super(GCSLToGym, self).__init__(env)
        self.observation_space = gym.spaces.Dict(
            dict(
                observation=env.observation_space,
                achieved_goal=env.goal_space,
                desired_goal=env.goal_space,
            ),
        )
        self.goal_threshold = goal_threshold

        self.current_state = None
        self.desired_goal_state = None
        self.desired_goal = None

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment."""
        self.desired_goal_state = self.env.sample_goal()
        self.desired_goal = self.env.extract_goal(self.desired_goal_state)
        return super(GCSLToGym, self).reset()

    def step(
        self,
        action: Union[int, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Union[int, np.int64, np.float64],
        np.bool_,
        Dict[str, Any],
    ]:
        """Take a step in the environment."""
        observation, reward, _, info = super(GCSLToGym, self).step(action)
        distance_to_goal = self.env.goal_distance(
            self.current_state,
            self.desired_goal_state,
        )
        done = distance_to_goal < self.goal_threshold
        return observation, reward, done, info

    def observation(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Fetch the environment observation."""
        self.current_state = state
        return dict(
            observation=self.env.observation(state),
            achieved_goal=self.env.extract_goal(state),
            desired_goal=self.desired_goal,
        )

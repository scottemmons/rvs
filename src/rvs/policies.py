"""Defines the policy class, including the main training step logic."""

from typing import Optional, Tuple, Type

import gym
from gym import spaces
import numpy as np
import pytorch_lightning as pl
from stable_baselines3.common import policies, type_aliases, utils
import torch
from torch import nn, optim

from rvs import dataset, layers, step, util


def make_obs_goal_space(
    observation_space: gym.Space,
    unconditional_policy: bool = False,
    reward_conditioning: bool = False,
    xy_conditioning: bool = False,
) -> gym.Space:
    """Create the policy's input space.

    This includes the observation as well as a possible goal.

    Args:
        observation_space: The observation space of the environment. By default, it's
            duplicated to create the goal space.
        unconditional_policy: If True, do not use any goals, and only use the
            observation_space.
        reward_conditioning: If True, condition on a reward scalar appended to the
            observation_space.
        xy_conditioning: If True, condition on (x, y) coordinates appended to the
            observation_space.

    Returns:
        The new space including observation and goal.

    Raises:
        ValueError: If conflicting types of spaces are specified.
    """
    if sum([unconditional_policy, reward_conditioning, xy_conditioning]) > 1:
        raise ValueError("You must choose at most one policy conditioning setting.")

    if unconditional_policy:
        return observation_space
    elif reward_conditioning:
        return util.add_scalar_to_space(observation_space)
    elif xy_conditioning:
        return util.add_scalar_to_space(util.add_scalar_to_space(observation_space))
    else:
        return util.create_observation_goal_space(observation_space)


class RvS(pl.LightningModule):
    """A Reinforcement Learning via Supervised Learning (RvS) policy."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_size: int = 1024,
        depth: int = 2,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        activation_fn: Type[nn.Module] = nn.ReLU,
        dropout_p: float = 0.1,
        unconditional_policy: bool = False,
        reward_conditioning: bool = False,
        env_name: Optional[str] = None,
    ):
        """Builds RvS.

        Args:
            observation_space: The policy's observation space
            action_space: The policy's action space
            hidden_size: The width of each hidden layer
            depth: The number of hidden layers
            learning_rate: A learning rate held constant throughout training
            batch_size: The batch size for each gradient step
            activation_fn: The network's activation function
            dropout_p: The dropout probability
            unconditional_policy: If True, ignore goals and act only based on
                observations
            reward_conditioning: If True, condition on a reward scalar instead of future
                observations
            env_name: The name of the environment for which to configure the policy
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.unconditional_policy = unconditional_policy
        self.batch_size = batch_size
        self.save_hyperparameters(
            "hidden_size",
            "depth",
            "learning_rate",
            "batch_size",
            "activation_fn",
            "dropout_p",
            "unconditional_policy",
            "reward_conditioning",
            "env_name",
        )

        xy_conditioning = (
            env_name in step.d4rl_antmaze
            and not unconditional_policy
            and not reward_conditioning
        )
        observation_goal_space = make_obs_goal_space(
            observation_space,
            unconditional_policy=unconditional_policy,
            reward_conditioning=reward_conditioning,
            xy_conditioning=xy_conditioning,
        )
        lr_schedule = utils.constant_fn(learning_rate)
        net_arch = [hidden_size] * depth
        layers.DropoutActivation.activation_fn = activation_fn
        layers.DropoutActivation.p = dropout_p
        self.model = ExtendedActorCriticPolicy(
            observation_goal_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=layers.DropoutActivation,
        )

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute a forward pass with the model."""
        return self.model.forward(*args, **kwargs)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        log_prefix: str = "train",
    ) -> torch.Tensor:
        """Computes loss for a training batch."""
        obs_goal, action = batch
        _, log_probs, _, prediction = self.model.evaluate_and_predict(
            obs_goal,
            action,
        )

        prob_true_act = torch.exp(log_probs).mean()
        loss = -log_probs.mean()

        self.log(f"{log_prefix}_prob_true_act", prob_true_act)
        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        try:
            self.log(f"{log_prefix}_std", torch.exp(self.model.log_std).mean())
            self.log(f"{log_prefix}_log_std", self.model.log_std.mean())
        except AttributeError:
            pass
        if prediction is not None:
            self.log(f"{log_prefix}_mse", ((prediction - action) ** 2).mean())

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Computes loss for a validation batch."""
        loss = self.training_step(batch, batch_idx, log_prefix="val")
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        """Configures the optimizer used by PyTorch Lightning."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ##################
    # POLICY FUNCTIONS
    ##################

    def get_probabilities(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        actions: np.ndarray,
    ) -> torch.Tensor:
        """Get the policy's probabilities.

        Returns a probability for each action given the corresponding observation and
        goal.
        """
        assert actions.shape[0] == observations.shape[0] == goals.shape[0]
        s_g_tensor = dataset.make_s_g_tensor(observations, goals)
        a_tensor = torch.tensor(actions)

        self.model.eval()
        with torch.no_grad():
            _, log_probs, _ = self.model.evaluate_actions(s_g_tensor, a_tensor)
        probs = torch.exp(log_probs)
        return probs

    def get_action(
        self,
        observation: np.ndarray,
        goal: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Get an action for a single observation / goal pair."""
        return self.get_actions(
            observation[np.newaxis],
            goal[np.newaxis],
            deterministic=deterministic,
        )[0]

    def get_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Get actions for each observation / goal pair."""
        assert observations.shape[0] == goals.shape[0]

        self.model.eval()
        with torch.no_grad():
            if self.unconditional_policy:
                s_tensor = torch.tensor(observations)
                actions = self.model._predict(s_tensor, deterministic=deterministic)
            else:
                s_g_tensor = dataset.make_s_g_tensor(observations, goals)
                actions = self.model._predict(s_g_tensor, deterministic=deterministic)

        return actions.cpu().numpy()


class ExtendedActorCriticPolicy(policies.ActorCriticPolicy):
    """Extends the functionality of stable-baseline3's ActorCriticPolicy.

    The extended functionality includes:
    - Action and value predictions at the same time as evaluating probabilities.
    - The option to skip value function computation.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: type_aliases.Schedule,
        compute_values: bool = False,
        **kwargs,
    ):
        """Builds ExtendedActorCriticPolicy.

        Args:
            observation_space: The policy's observation space.
            action_space: The policy's action space.
            lr_schedule: A learning rate schedule.
            compute_values: We'll skip value function computation unless this is True.
            **kwargs: Keyword arguments passed along to parent class.
        """
        self.compute_values = compute_values

        super(ExtendedActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    def _build(self, lr_schedule: type_aliases.Schedule) -> None:
        super(ExtendedActorCriticPolicy, self)._build(lr_schedule)
        # Skip unused value function computation
        if not self.compute_values:
            self.value_net = nn.Sequential()  # Identity function
            self.optimizer = self.optimizer_class(
                self.parameters(),
                lr=lr_schedule(1),
                **self.optimizer_kwargs,
            )

    def evaluate_and_predict(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate probability of actions and provide action and value prediction."""
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        if isinstance(self.action_space, spaces.Box):
            predictions = distribution.get_actions(deterministic=True)
        else:
            predictions = None
        return values, log_prob, distribution.entropy(), predictions

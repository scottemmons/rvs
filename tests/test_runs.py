"""Smoke tests for training runs."""

import os
import subprocess

import pytest

kitchen_command = [
    "python",
    "src/rvs/train.py",
    "--run_tag",
    "d4rl-test-run",
    "--seed",
    "0",
    "--dropout_p",
    "0.1",
    "--hidden_size",
    "4",
    "--env_name",
    "kitchen-complete-v0",
    "--val_frac",
    "0.1",
    "--learning_rate",
    "1e-3",
    "--batch_size",
    "2",
    "--max_steps",
    "20",
    "--checkpoint_every_n_steps",
    "10",
    "--depth",
    "2",
    "--analyze_d4rl",
    "--trajectory_samples",
    "2",
    "--d4rl_analysis",
    "kitchen_subtasks",
]
gym_command = [
    "python",
    "src/rvs/train.py",
    "--run_tag",
    "d4rl-test-run",
    "--seed",
    "0",
    "--dropout_p",
    "0",
    "--hidden_size",
    "4",
    "--env_name",
    "halfcheetah-medium-expert-v2",
    "--reward_conditioning",
    "--val_frac",
    "0",
    "--learning_rate",
    "1e-3",
    "--batch_size",
    "16384",
    "--epochs",
    "2",
    "--checkpoint_every_n_epochs",
    "1",
    "--depth",
    "2",
    "--analyze_d4rl",
    "--trajectory_samples",
    "2",
]
antmaze_command = [
    "python",
    "src/rvs/train.py",
    "--run_tag",
    "d4rl-test-run",
    "--seed",
    "0",
    "--dropout_p",
    "0.1",
    "--hidden_size",
    "4",
    "--env_name",
    "antmaze-umaze-diverse-v2",
    "--val_frac",
    "0.1",
    "--learning_rate",
    "1e-3",
    "--batch_size",
    "2",
    "--max_steps",
    "20",
    "--checkpoint_every_n_steps",
    "10",
    "--depth",
    "2",
    "--analyze_d4rl",
    "--trajectory_samples",
    "2",
]
lunar_command = [
    "python",
    "src/rvs/train.py",
    "--val_frac",
    "0",
    "--run_tag",
    "gcsl-test-run",
    "--seed",
    "0",
    "--dropout_p",
    "0",
    "--hidden_size",
    "4",
    "--env_name",
    "lunar",
    "--total_steps",
    "200",
    "--learning_rate",
    "1e-3",
    "--batch_size",
    "1000",
    "--epochs",
    "6",
    "--checkpoint_every_n_epochs",
    "2",
    "--depth",
    "2",
    "--visualize",
    "--trajectory_samples",
    "2",
]


@pytest.mark.expensive
def test_kitchen_run():
    """Check that a D4RL kitchen run completes with no errors."""
    os.environ["WANDB_MODE"] = "offline"
    subprocess.run(kitchen_command, check=True)


@pytest.mark.expensive
def test_gym_run():
    """Check that a D4RL gym run completes with no errors."""
    os.environ["WANDB_MODE"] = "offline"
    subprocess.run(gym_command, check=True)


@pytest.mark.expensive
def test_antmaze_run():
    """Check that a D4RL antmaze run completes with no errors."""
    os.environ["WANDB_MODE"] = "offline"
    subprocess.run(antmaze_command, check=True)


@pytest.mark.expensive
def test_gcsl_run():
    """Check that a GCSL lunar run completes with no errors."""
    os.environ["WANDB_MODE"] = "offline"
    subprocess.run(lunar_command, check=True)

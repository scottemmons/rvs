"""Definitions of neural network layers."""

import torch
from torch import nn


class DropoutActivation(nn.Module):
    """Combines dropout and activation into a single module.

    This is useful for adding dropout to a Stable Baselines3 policy, which takes an
    activation function as input.
    """

    activation_fn = nn.ReLU
    p = 0.1

    def __init__(self):
        """Instantiate the dropout and activation layers."""
        super(DropoutActivation, self).__init__()
        self.activation = DropoutActivation.activation_fn()
        self.dropout = nn.Dropout(p=DropoutActivation.p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass: activation function first, then dropout."""
        return self.dropout(self.activation(x))

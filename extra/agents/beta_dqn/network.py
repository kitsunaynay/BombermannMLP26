from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .kit.actions import N_ACTIONS

MASK_FILL = -1e8


def orthogonal_init(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    """Initialize a linear or convolutional layer."""
    nn.init.orthogonal_(layer.weight, gain=gain)

    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)

    return layer


class QNetwork(nn.Module):
    """CNN that predicts one Q-value for each Bomberman action."""

    def __init__(
        self,
        in_channels: int,
        spatial_size: int,
        channels: Sequence[int] = (32, 64, 64),
        hidden_dim: int = 256,
        n_actions: int = N_ACTIONS,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.n_actions = n_actions

        # Same convolutional feature extractor as the PPO agent
        layers = []
        previous = in_channels

        for width in channels:
            layers.append(
                orthogonal_init(
                    nn.Conv2d(previous, width, kernel_size=3, padding=1)
                )
            )
            layers.append(nn.ReLU(inplace=True))
            previous = width

        self.backbone = nn.Sequential(*layers)

        # Determine the flattened CNN output size automatically.
        with torch.no_grad():
            probe = torch.zeros(
                1, in_channels, spatial_size, spatial_size
            )
            self.flat_dim = int(
                self.backbone(probe).flatten(1).shape[1]
            )

        self.trunk = nn.Sequential(
            orthogonal_init(
                nn.Linear(self.flat_dim, hidden_dim)
            ),
            nn.ReLU(inplace=True),
        )

        # DQN has one output for each action.
        self.q_head = orthogonal_init(
            nn.Linear(hidden_dim, n_actions),
            gain=1.0,
        )

    def forward(
        self,
        observation: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return Q(s, a) for every action."""

        hidden = self.trunk(
            self.backbone(observation).flatten(1)
        )

        q_values = self.q_head(hidden)

        # Unsafe/invalid actions cannot be selected by argmax.
        if action_mask is not None:
            usable = action_mask.any(dim=-1, keepdim=True)

            q_values = torch.where(
                action_mask | ~usable,
                q_values,
                torch.full_like(q_values, MASK_FILL),
            )

        return q_values

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

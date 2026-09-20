from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .kit.actions import N_ACTIONS

MASK_FILL = -1e8


def orthogonal_init(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    # three-layer conv backbone + mlp heads
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

        # convolution backbone
        layers = []
        previous = in_channels
        for width in channels:
            layers.append(orthogonal_init(nn.Conv2d(previous, width, 3, padding=1)))
            layers.append(nn.ReLU(inplace=True))
            previous = width
        self.backbone = nn.Sequential(*layers)

        # probe to find flattened size
        with torch.no_grad():
            probe = torch.zeros(1, in_channels, spatial_size, spatial_size)
            self.flat_dim = int(self.backbone(probe).flatten(1).shape[1])

        # shared trunk + separate heads
        self.trunk = nn.Sequential(
            orthogonal_init(nn.Linear(self.flat_dim, hidden_dim)),
            nn.ReLU(inplace=True),
        )
        self.policy_head = orthogonal_init(nn.Linear(hidden_dim, n_actions), gain=0.01)
        self.value_head = orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(
        self,
        observation: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # shared feature extraction
        hidden = self.trunk(self.backbone(observation).flatten(1))
        logits = self.policy_head(hidden)

        # mask invalid actions
        if action_mask is not None:
            usable = action_mask.any(dim=-1, keepdim=True)
            logits = torch.where(action_mask | ~usable, logits, torch.full_like(logits, MASK_FILL))

        return logits, self.value_head(hidden).squeeze(-1)

    @torch.inference_mode()
    def act(
        self,
        observation: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(observation, action_mask)
        distribution = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = distribution.sample()

        return action, distribution.log_prob(action), value

    def evaluate(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(observation, action_mask)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(action), distribution.entropy(), value

    def config_dict(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "spatial_size": self.spatial_size,
            "n_actions": self.n_actions,
            "flat_dim": self.flat_dim,
        }

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_network(config, in_channels: int) -> ActorCritic:
    return ActorCritic(
        in_channels=in_channels,
        spatial_size=config.spatial_size,
        channels=tuple(config.channels),
        hidden_dim=config.hidden_dim,
    )

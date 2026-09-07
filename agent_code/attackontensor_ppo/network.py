"""Actor-critic network.

Sized for the tournament budget rather than maximum capacity: one CPU thread
and 0.5 s per decision. An agent that overruns has its action replaced by
``WAIT`` and loses the excess from its next step's budget
(environment.py:448).

Three 3x3 convolutions preserving spatial extent, then a shared dense trunk that
forks into a policy head and a value head::

    (13, H, W) -> Conv3x3(32) -> ReLU
               -> Conv3x3(64) -> ReLU
               -> Conv3x3(64) -> ReLU
               -> flatten -> Linear(256) -> ReLU
                              |-> Linear(6)  policy logits
                              |-> Linear(1)  state value

On the full 17x17 board that is roughly 17M multiply-accumulates per forward
pass, single-digit milliseconds single-threaded. ``tools/latency_check.py``
measures it rather than trusting the arithmetic.

Initialisation is the usual PPO recipe: orthogonal weights, policy head scaled
down by 100x so the initial policy is near-uniform. Confident-but-arbitrary
starting logits collapse entropy before any learning happens.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .kit.actions import N_ACTIONS

#: Additive mask value for forbidden actions. Large and negative, but finite --
#: real -inf produces NaN gradients if a whole row is ever masked out.
MASK_FILL = -1e8


def orthogonal_init(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """Shared convolutional trunk with separate policy and value heads."""

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

        layers = []
        previous = in_channels
        for width in channels:
            layers.append(orthogonal_init(nn.Conv2d(previous, width, 3, padding=1)))
            layers.append(nn.ReLU(inplace=True))
            previous = width
        self.backbone = nn.Sequential(*layers)

        # Derived by a dummy forward rather than hand-computed, so changing the
        # observation mode or channel widths cannot silently desync the shapes.
        with torch.no_grad():
            probe = torch.zeros(1, in_channels, spatial_size, spatial_size)
            self.flat_dim = int(self.backbone(probe).flatten(1).shape[1])

        self.trunk = nn.Sequential(
            orthogonal_init(nn.Linear(self.flat_dim, hidden_dim)),
            nn.ReLU(inplace=True),
        )
        # gain 0.01: start near-uniform so entropy does not collapse at step 0.
        self.policy_head = orthogonal_init(nn.Linear(hidden_dim, n_actions), gain=0.01)
        self.value_head = orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0)

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        observation: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, value)``; masked actions get a large negative logit."""
        hidden = self.trunk(self.backbone(observation).flatten(1))
        logits = self.policy_head(hidden)

        if action_mask is not None:
            # Never mask an entire row: if every action is forbidden the agent
            # is dead anyway, and an all -inf row makes softmax produce NaN.
            usable = action_mask.any(dim=-1, keepdim=True)
            logits = torch.where(action_mask | ~usable, logits, torch.full_like(logits, MASK_FILL))

        return logits, self.value_head(hidden).squeeze(-1)

    # -- acting -------------------------------------------------------------
    @torch.inference_mode()
    def act(
        self,
        observation: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or take the mode of) the policy. Returns action, logprob, value."""
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
        """Log-probability, entropy and value for stored actions during an update."""
        logits, value = self.forward(observation, action_mask)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(action), distribution.entropy(), value

    # -- persistence --------------------------------------------------------
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
    """Construct the network described by a :class:`PPOConfig`."""
    return ActorCritic(
        in_channels=in_channels,
        spatial_size=config.spatial_size,
        channels=tuple(config.channels),
        hidden_dim=config.hidden_dim,
    )

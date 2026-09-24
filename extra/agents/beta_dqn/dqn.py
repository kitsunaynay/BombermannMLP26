from __future__ import annotations

import torch
import torch.nn.functional as F

from .network import QNetwork


def double_dqn_loss(
    online_network: QNetwork,
    target_network: QNetwork,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    next_masks: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute the Double DQN loss for one minibatch."""

    # Q-values predicted by the online network for the current states
    q_values = online_network(states)

    # Keep only Q(s, a) for the actions that were actually taken
    chosen_q_values = q_values.gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)

    with torch.no_grad():
        # Double DQN:
        # 1. online network chooses the best next action
        online_next_q = online_network(
            next_states,
            action_mask=next_masks,
        )

        next_actions = online_next_q.argmax(dim=1)

        # 2. target network evaluates that chosen action
        target_next_q = target_network(
            next_states,
            action_mask=next_masks,
        )

        next_q_values = target_next_q.gather(
            1, next_actions.unsqueeze(1)
        ).squeeze(1)

        targets = rewards + gamma * (1.0 - dones) * next_q_values

    return F.smooth_l1_loss(chosen_q_values, targets)


def update_target_network(
    online_network: QNetwork,
    target_network: QNetwork,
) -> None:
    """Copy the online-network parameters to the target network"""

    target_network.load_state_dict(online_network.state_dict())

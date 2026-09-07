"""PPO: rollout storage, generalised advantage estimation, and the update.

Proximal Policy Optimization (Schulman et al., 2017) maximises a clipped
surrogate objective

.. math::
   L^{CLIP} = \\mathbb{E}_t\\Big[\\min\\big(\\rho_t \\hat A_t,\\;
              \\mathrm{clip}(\\rho_t, 1-\\epsilon, 1+\\epsilon)\\hat A_t\\big)\\Big],
   \\qquad \\rho_t = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{old}}(a_t|s_t)}

The clip allows several gradient epochs per batch: once the new policy moves far
enough that :math:`\\rho_t` leaves the trust region, the objective flattens.
Advantages come from GAE (Schulman et al., 2015),

.. math::
   \\hat A_t = \\sum_{l \\ge 0} (\\gamma\\lambda)^l \\delta_{t+l},
   \\qquad \\delta_t = r_t + \\gamma V(s_{t+1})(1 - d_t) - V(s_t)

which trades bias against variance through :math:`\\lambda`.

The total loss adds a value term and an entropy bonus:

.. math::
   L = -L^{CLIP} + c_v L^{VF} - c_e \\mathcal{H}[\\pi]

Entropy collapse is the main failure mode here: an agent that becomes certain
too early stops discovering that bombs kill crates. Entropy is logged every
update as one of the failure detectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .kit.actions import N_ACTIONS


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation over one contiguous rollout.

    ``dones[t]`` marks the transition *out of* step ``t`` as terminal, which cuts
    both the bootstrap and the recursion so episodes never bleed into each other.

    Returns ``(advantages, returns)`` with ``returns = advantages + values``.
    """
    steps = len(rewards)
    advantages = np.zeros(steps, dtype=np.float64)

    running = 0.0
    next_value = float(last_value)
    for t in reversed(range(steps)):
        non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        running = delta + gamma * lam * non_terminal * running
        advantages[t] = running
        next_value = values[t]

    return advantages, advantages + values


class RolloutBuffer:
    """Fixed-capacity storage for on-policy experience.

    Pre-allocated numpy arrays rather than a list of tuples: rollouts are
    collected step by step but consumed as one big batch, and repeatedly
    re-stacking a Python list of observations dominates the update cost.
    """

    def __init__(self, capacity: int, observation_shape: Tuple[int, ...], n_actions: int = N_ACTIONS):
        self.capacity = int(capacity)
        self.observation_shape = tuple(observation_shape)

        self.observations = np.zeros((self.capacity, *self.observation_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=bool)
        self.masks = np.ones((self.capacity, n_actions), dtype=bool)

        self.size = 0

    def __len__(self) -> int:
        return self.size

    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity

    def add(
        self,
        observation: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        if self.is_full:
            raise RuntimeError("RolloutBuffer is full; compute advantages and reset first")

        index = self.size
        self.observations[index] = observation
        self.actions[index] = action
        self.log_probs[index] = log_prob
        self.rewards[index] = reward
        self.values[index] = value
        self.dones[index] = done
        if mask is not None:
            self.masks[index] = mask
        else:
            self.masks[index] = True
        self.size += 1

    def reset(self) -> None:
        self.size = 0

    def compute_advantages(
        self, last_value: float, gamma: float, lam: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        return compute_gae(
            self.rewards[: self.size],
            self.values[: self.size],
            self.dones[: self.size],
            last_value,
            gamma,
            lam,
        )


@dataclass
class UpdateMetrics:
    """Diagnostics from one PPO update, all of them worth plotting.

    ``approx_kl`` and ``clip_fraction`` say whether the trust region is doing
    anything; ``entropy`` catches premature determinism; ``explained_variance``
    says whether the critic is learning at all (values near or below zero mean
    it predicts no better than the mean return).
    """

    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    grad_norm: float = 0.0
    n_updates: int = 0
    early_stopped: bool = False

    def as_dict(self) -> Dict[str, float]:
        return {
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "approx_kl": self.approx_kl,
            "clip_fraction": self.clip_fraction,
            "explained_variance": self.explained_variance,
            "grad_norm": self.grad_norm,
            "n_updates": float(self.n_updates),
            "early_stopped": float(self.early_stopped),
        }


def explained_variance(predictions: np.ndarray, targets: np.ndarray) -> float:
    """1 - Var(target - prediction) / Var(target); 0 means "no better than the mean"."""
    variance = float(np.var(targets))
    if variance < 1e-12:
        return 0.0
    return float(1.0 - np.var(targets - predictions) / variance)


class PPOLearner:
    """Owns the network, the optimiser, and the update step."""

    def __init__(self, network: nn.Module, config, device: str = "cpu"):
        self.network = network
        self.config = config
        self.device = torch.device(device)
        self.network.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self.progress = 0.0  # 0 -> 1 over training, drives the anneal schedules

    # -- schedules ----------------------------------------------------------
    def _annealed(self, start: float, final: float) -> float:
        if not self.config.anneal_schedules:
            return start
        return start + (final - start) * min(max(self.progress, 0.0), 1.0)

    @property
    def learning_rate(self) -> float:
        return self._annealed(self.config.learning_rate, self.config.learning_rate_final)

    @property
    def entropy_coefficient(self) -> float:
        return self._annealed(
            self.config.entropy_coefficient, self.config.entropy_coefficient_final
        )

    # -- update -------------------------------------------------------------
    def update(
        self,
        buffer: RolloutBuffer,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> UpdateMetrics:
        config = self.config
        count = len(buffer)
        if count == 0:
            return UpdateMetrics()

        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

        observations = torch.as_tensor(buffer.observations[:count], device=self.device)
        actions = torch.as_tensor(buffer.actions[:count], device=self.device)
        old_log_probs = torch.as_tensor(buffer.log_probs[:count], device=self.device)
        old_values = torch.as_tensor(buffer.values[:count], device=self.device)
        masks = torch.as_tensor(buffer.masks[:count], device=self.device)
        advantage_tensor = torch.as_tensor(advantages.astype(np.float32), device=self.device)
        return_tensor = torch.as_tensor(returns.astype(np.float32), device=self.device)

        metrics = UpdateMetrics()
        entropy_coefficient = self.entropy_coefficient
        batch_size = min(config.minibatch_size, count)
        indices = np.arange(count)
        n_updates = 0

        for _ in range(config.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, count, batch_size):
                batch = indices[start : start + batch_size]
                if len(batch) < 2:
                    continue  # a 1-sample batch makes advantage normalisation undefined
                batch_index = torch.as_tensor(batch, device=self.device)

                batch_advantages = advantage_tensor[batch_index]
                if config.normalise_advantages:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                        batch_advantages.std() + 1e-8
                    )

                log_probs, entropy, values = self.network.evaluate(
                    observations[batch_index], actions[batch_index], masks[batch_index]
                )

                log_ratio = log_probs - old_log_probs[batch_index]
                ratio = log_ratio.exp()

                surrogate_1 = ratio * batch_advantages
                surrogate_2 = (
                    torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                if config.clip_value_loss:
                    # Clipping the value update too keeps the critic from
                    # lurching on a single batch of high-variance returns.
                    clipped = old_values[batch_index] + torch.clamp(
                        values - old_values[batch_index],
                        -config.clip_epsilon,
                        config.clip_epsilon,
                    )
                    value_loss = 0.5 * torch.max(
                        (values - return_tensor[batch_index]).pow(2),
                        (clipped - return_tensor[batch_index]).pow(2),
                    ).mean()
                else:
                    value_loss = 0.5 * (values - return_tensor[batch_index]).pow(2).mean()

                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    - entropy_coefficient * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), config.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    # Schulman's low-variance KL estimator; always non-negative.
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_fraction = (
                        ((ratio - 1.0).abs() > config.clip_epsilon).float().mean().item()
                    )

                metrics.policy_loss += policy_loss.item()
                metrics.value_loss += value_loss.item()
                metrics.entropy += entropy_mean.item()
                metrics.approx_kl += approx_kl
                metrics.clip_fraction += clip_fraction
                metrics.grad_norm += float(grad_norm)
                n_updates += 1

                if config.target_kl > 0 and approx_kl > config.target_kl:
                    # The policy has moved too far for this batch to stay
                    # on-policy enough to trust; stop rather than keep pushing.
                    metrics.early_stopped = True
                    break

            if metrics.early_stopped:
                break

        if n_updates:
            metrics.policy_loss /= n_updates
            metrics.value_loss /= n_updates
            metrics.entropy /= n_updates
            metrics.approx_kl /= n_updates
            metrics.clip_fraction /= n_updates
            metrics.grad_norm /= n_updates
        metrics.n_updates = n_updates

        with torch.no_grad():
            predictions = self.network.evaluate(observations, actions, masks)[2]
        metrics.explained_variance = explained_variance(
            predictions.cpu().numpy(), returns.astype(np.float32)
        )

        return metrics

    # -- persistence --------------------------------------------------------
    def save(self, path, extra: Optional[dict] = None) -> None:
        """Atomically write a checkpoint.

        Same reasoning as the Q-table: training gets interrupted, and a
        half-written ``.pt`` would be indistinguishable from a good one until
        the tournament tried to load it.
        """
        import os
        import tempfile
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "state_dict": self.network.state_dict(),
            "network": self.network.config_dict(),
            "config": self.config.to_dict(),
            "progress": self.progress,
        }
        if extra:
            payload.update(extra)

        handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(handle)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

"""Picklable factories for worker processes."""

from __future__ import annotations

from typing import Callable, Optional


def make_ppo_reward_fn() -> Callable:
    """Reward function matching the PPO agent's own configuration."""
    from agent_code.attackontensor_ppo import rewards as R
    from agent_code.attackontensor_ppo.config import PPOConfig

    config = PPOConfig.load()

    def reward_fn(old_state, action, new_state, events) -> float:
        reward, _ = R.compute_reward(old_state, action, new_state, events, config)
        return float(reward)

    return reward_fn


def make_ppo_transform() -> Callable:
    """Observation encoder, run inside the worker.

    Encoding in the child parallelises the work *and* shrinks what crosses the
    pipe: a float32 tensor instead of a whole game-state dictionary with its
    numpy arrays and Python lists.
    """
    from agent_code.attackontensor_ppo import tensorizer as T
    from agent_code.attackontensor_ppo.config import PPOConfig

    config = PPOConfig.load()

    def transform(game_state):
        return {
            "observation": T.state_to_tensor(game_state, config),
            "mask": T.action_mask_for(game_state, config),
        }

    return transform


def make_ql_reward_fn() -> Callable:
    """Reward function matching the Q-learning agent's configuration."""
    from agent_code.attackontensor_ql import rewards as R
    from agent_code.attackontensor_ql.config import QLConfig

    config = QLConfig.load()

    def reward_fn(old_state, action, new_state, events) -> float:
        reward, _ = R.compute_reward(old_state, action, new_state, events, config)
        return float(reward)

    return reward_fn

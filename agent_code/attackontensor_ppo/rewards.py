from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .config import PPOConfig
from .kit.rewards import (
    CUSTOM_EVENTS,
    detect_custom_events,
    potential as _potential,
    reward_from_events as _reward_from_events,
    shaping_term as _shaping_term,
)


def potential(game_state: Optional[dict], config: PPOConfig) -> float:
    return _potential(
        game_state,
        config.potential_coin,
        config.potential_crate,
        config.potential_danger,
    )


def shaping_term(
    old_game_state: Optional[dict],
    new_game_state: Optional[dict],
    config: PPOConfig,
) -> float:
    if not config.use_potential_shaping:
        return 0.0
    return _shaping_term(
        old_game_state,
        new_game_state,
        config.gamma,
        config.potential_coin,
        config.potential_crate,
        config.potential_danger,
    )


def reward_from_events(events: Sequence[str], config: PPOConfig) -> float:
    return _reward_from_events(events, config.event_rewards)


def compute_reward(
    old_game_state: Optional[dict],
    self_action: Optional[str],
    new_game_state: Optional[dict],
    events: Sequence[str],
    config: PPOConfig,
) -> Tuple[float, List[str]]:
    all_events = list(events)

    if config.use_custom_events:
        all_events.extend(
            detect_custom_events(old_game_state, self_action, new_game_state, events)
        )

    reward = _reward_from_events(all_events, config.event_rewards)
    reward += shaping_term(old_game_state, new_game_state, config)
    return reward, all_events

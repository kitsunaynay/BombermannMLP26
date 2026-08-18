"""Reward assembly for the Q-learning agent.

A thin adapter: the mechanics (potential function, custom-event detection) live
in :mod:`kit.rewards` so both agents share one implementation, while the weights
and the on/off switches stay in this agent's own config. See the kit module for
the Ng et al. (1999) policy-invariance argument behind the shaping term.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .config import QLConfig
from .kit.rewards import (  # noqa: F401 - re-exported for tests and diagnostics
    CUSTOM_EVENTS,
    ENTERED_DANGER,
    ESCAPED_DANGER,
    MOVED_AWAY_FROM_COIN,
    MOVED_TOWARD_COIN,
    SUICIDAL_BOMB,
    SURVIVED_STEP,
    USEFUL_BOMB,
    USELESS_BOMB,
    WAITED_IN_DANGER,
    detect_custom_events,
    potential as _potential,
    reward_from_events as _reward_from_events,
    shaping_term as _shaping_term,
)


def potential(game_state: Optional[dict], config: QLConfig) -> float:
    """:math:`\\Phi(s)` under this agent's coefficients."""
    return _potential(
        game_state,
        config.potential_coin,
        config.potential_crate,
        config.potential_danger,
    )


def shaping_term(
    old_game_state: Optional[dict],
    new_game_state: Optional[dict],
    config: QLConfig,
) -> float:
    """:math:`\\gamma \\Phi(s') - \\Phi(s)`, or zero when shaping is disabled."""
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


def reward_from_events(events: Sequence[str], config: QLConfig) -> float:
    return _reward_from_events(events, config.event_rewards)


def compute_reward(
    old_game_state: Optional[dict],
    self_action: Optional[str],
    new_game_state: Optional[dict],
    events: Sequence[str],
    config: QLConfig,
) -> Tuple[float, List[str]]:
    """Total reward for one transition, plus the events actually credited."""
    all_events = list(events)

    if config.use_custom_events:
        all_events.extend(
            detect_custom_events(old_game_state, self_action, new_game_state, events)
        )

    reward = _reward_from_events(all_events, config.event_rewards)
    reward += shaping_term(old_game_state, new_game_state, config)
    return reward, all_events

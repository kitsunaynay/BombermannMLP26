from __future__ import annotations

from typing import Optional

import numpy as np

import settings as s

from .config import PPOConfig
from .kit import geometry as G
from .kit import pathfind as P
from .kit.actions import ACTION_DELTAS, MOVE_ACTIONS, WAIT

# board encoded as multi-plane tensor: walls, dangers, coins, positions, etc.
# lets convolutions find their own features instead of hand-picked numbers
# channels 0-12 always present; 13-16 optional (survival_channels)
BASE_CHANNELS = 13
SURVIVAL_CHANNELS = 4
N_CHANNELS = BASE_CHANNELS + SURVIVAL_CHANNELS

(
    CH_WALL,
    CH_CRATE,
    CH_FREE,
    CH_SELF,
    CH_OTHERS,
    CH_BOMB_READY,
    CH_BOMB,
    CH_BOMB_TIMER,
    CH_DANGER,
    CH_EXPLOSION,
    CH_COIN,
    CH_OTHERS_BOMB_READY,
    CH_PROGRESS,
    CH_SURVIVAL_DURATION,
    CH_SURVIVAL_BREADTH,
    CH_BLAST_MARGIN,
    CH_CONTESTED,
) = range(N_CHANNELS)

CHANNEL_NAMES = (
    "wall",
    "crate",
    "free",
    "self",
    "others",
    "bomb_ready",
    "bomb",
    "bomb_timer",
    "danger",
    "explosion",
    "coin",
    "others_bomb_ready",
    "progress",
    "survival_duration",
    "survival_breadth",
    "blast_margin",
    "contested",
)

_MARGIN_SCALE = 5.0


def state_to_tensor(
    game_state: Optional[dict],
    config: Optional[PPOConfig] = None,
) -> Optional[np.ndarray]:
    # encode game state as multi-plane tensor
    if game_state is None:
        return None

    config = config or PPOConfig.load()

    # unpack state
    field = game_state["field"]
    _, _, bomb_available, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    others = [xy for (_, _, _, xy) in game_state["others"]]
    others_ready = [xy for (_, _, ready, xy) in game_state["others"] if ready]
    coins = game_state["coins"]
    explosion_map = np.asarray(game_state["explosion_map"])

    tensor = np.zeros((n_channels(config), *field.shape), dtype=np.float32)

    # static structures
    tensor[CH_WALL] = field == G.WALL
    tensor[CH_CRATE] = field == G.CRATE
    tensor[CH_FREE] = field == G.FREE

    # agents
    tensor[CH_SELF, x, y] = 1.0
    for ox, oy in others:
        tensor[CH_OTHERS, ox, oy] = 1.0
    for ox, oy in others_ready:
        tensor[CH_OTHERS_BOMB_READY, ox, oy] = 1.0

    # bomb state
    tensor[CH_BOMB_READY] = 1.0 if bomb_available else 0.0

    timer_scale = float(s.BOMB_TIMER)
    for (bx, by), timer in bombs:
        tensor[CH_BOMB, bx, by] = 1.0
        tensor[CH_BOMB_TIMER, bx, by] = (timer_scale - timer) / timer_scale

    # danger map
    danger = G.danger_map(field, bombs, explosion_map)
    threatened = danger != G.SAFE
    tensor[CH_DANGER][threatened] = (timer_scale - danger[threatened]) / timer_scale
    tensor[CH_DANGER] = np.clip(tensor[CH_DANGER], 0.0, 1.0)

    tensor[CH_EXPLOSION] = (explosion_map >= 1).astype(np.float32)

    # collectibles
    for cx, cy in coins:
        tensor[CH_COIN, cx, cy] = 1.0

    tensor[CH_PROGRESS] = min(game_state["step"] / s.MAX_STEPS, 1.0)

    # optional survival channels
    if config.survival_channels:
        _fill_survival_channels(tensor, field, (x, y), bombs, others, danger)

    # egocentric crop if configured
    if config.observation == "ego":
        tensor = egocentric_crop(tensor, x, y, config.ego_radius)

    return tensor


def _fill_survival_channels(tensor, field, position, bombs, others, danger) -> None:
    # compute escape routes and reachability constraints
    x, y = position
    passable = G.free_mask(field, bombs, others)
    opponent_reachable = P.opponent_reachability(
        field, others, horizon=P.SURVIVAL_HORIZON, origin=(x, y)
    )
    for action in (*MOVE_ACTIONS, WAIT):
        dx, dy = ACTION_DELTAS[action]
        nx, ny = x + dx, y + dy
        if not G.in_bounds(field, nx, ny):
            continue
        profile = P.survival_profile(
            field, (x, y), action, danger, passable,
            opponent_reachable=opponent_reachable,
        )
        tensor[CH_SURVIVAL_DURATION, nx, ny] = min(1.0, profile.duration / P.SURVIVAL_HORIZON)
        tensor[CH_SURVIVAL_BREADTH, nx, ny] = profile.breadth
        tensor[CH_BLAST_MARGIN, nx, ny] = min(1.0, profile.min_margin / _MARGIN_SCALE)
        tensor[CH_CONTESTED, nx, ny] = profile.contested


def egocentric_crop(tensor: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    # center view on agent position with wall padding
    channels = tensor.shape[0]
    padded = np.zeros(
        (channels, tensor.shape[1] + 2 * radius, tensor.shape[2] + 2 * radius),
        dtype=tensor.dtype,
    )
    padded[CH_WALL] = 1.0
    padded[:, radius:-radius, radius:-radius] = tensor

    return np.ascontiguousarray(
        padded[:, x : x + 2 * radius + 1, y : y + 2 * radius + 1]
    )


def n_channels(config: PPOConfig) -> int:
    return BASE_CHANNELS + (SURVIVAL_CHANNELS if config.survival_channels else 0)


def observation_shape(config: PPOConfig) -> tuple:
    size = config.spatial_size
    return (n_channels(config), size, size)


def action_mask_for(game_state: dict, config: PPOConfig) -> np.ndarray:
    from .kit import safety

    field, position, bomb_available, _, _, _, danger, passable = P.game_state_context(
        game_state
    )
    return safety.action_mask(
        field, position, danger, passable, bomb_available,
        mode=config.safety_mode, bomb_gate=config.bomb_gate,
    )

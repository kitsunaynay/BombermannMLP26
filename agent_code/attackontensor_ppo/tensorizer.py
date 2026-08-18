"""Multi-channel spatial encoding of the game state.

The Q-learning agent compresses the board into a dozen hand-designed integers.
This agent does the opposite: it hands the network an almost-raw spatial picture
and lets the convolutions discover their own features, which is the approach the
brief describes as "very powerful ... especially when it automatically learns
the features".

Layout is ``(C, X, Y)`` with ``C = 13``, indexed the same way as
``game_state['field']``. Note the board is stored in image coordinates, so the
array is the transpose of what the GUI shows -- the brief flags this too. Conv2d
does not care which axis means what as long as the convention never changes, and
keeping ``[x, y]`` throughout means channels can be filled straight from the
state dictionary without transposing.

===  ====================================================================
Ch   Contents
===  ====================================================================
0    stone walls
1    crates
2    free floor
3    this agent's position
4    opponents' positions
5    our bomb is available (constant plane)
6    a bomb occupies this tile
7    bomb timer, normalised, larger = more imminent
8    danger: steps until this tile is engulfed, normalised
9    an explosion is lethal here right now
10   collectable coins
11   opponents that still have a bomb in hand
12   round progress, step / MAX_STEPS (constant plane)
===  ====================================================================

Channels 8 and 9 are the ones that matter most: they hand the network the
timing information that :mod:`kit.geometry` derives, rather than making it
rediscover blast propagation from raw bomb positions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import settings as s

from .config import PPOConfig
from .kit import geometry as G
from .kit import pathfind as P

N_CHANNELS = 13

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
)


def state_to_tensor(
    game_state: Optional[dict],
    config: Optional[PPOConfig] = None,
) -> Optional[np.ndarray]:
    """Encode a game state as a ``(C, H, W)`` float32 array.

    Returns ``None`` for a missing state; the framework passes ``None`` for dead
    agents (environment.py:397).
    """
    if game_state is None:
        return None

    config = config or PPOConfig.load()

    field = game_state["field"]
    _, _, bomb_available, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    others = [xy for (_, _, _, xy) in game_state["others"]]
    others_ready = [xy for (_, _, ready, xy) in game_state["others"] if ready]
    coins = game_state["coins"]
    explosion_map = np.asarray(game_state["explosion_map"])

    tensor = np.zeros((N_CHANNELS, *field.shape), dtype=np.float32)

    tensor[CH_WALL] = field == G.WALL
    tensor[CH_CRATE] = field == G.CRATE
    tensor[CH_FREE] = field == G.FREE

    tensor[CH_SELF, x, y] = 1.0
    for ox, oy in others:
        tensor[CH_OTHERS, ox, oy] = 1.0
    for ox, oy in others_ready:
        tensor[CH_OTHERS_BOMB_READY, ox, oy] = 1.0

    tensor[CH_BOMB_READY] = 1.0 if bomb_available else 0.0

    timer_scale = float(s.BOMB_TIMER)
    for (bx, by), timer in bombs:
        tensor[CH_BOMB, bx, by] = 1.0
        # Invert so an imminent detonation is a *large* activation.
        tensor[CH_BOMB_TIMER, bx, by] = (timer_scale - timer) / timer_scale

    danger = G.danger_map(field, bombs, explosion_map)
    threatened = danger != G.SAFE
    # Same inversion: 1.0 means "lethal at the end of this step".
    tensor[CH_DANGER][threatened] = (timer_scale - danger[threatened]) / timer_scale
    tensor[CH_DANGER] = np.clip(tensor[CH_DANGER], 0.0, 1.0)

    tensor[CH_EXPLOSION] = (explosion_map >= 1).astype(np.float32)

    for cx, cy in coins:
        tensor[CH_COIN, cx, cy] = 1.0

    tensor[CH_PROGRESS] = min(game_state["step"] / s.MAX_STEPS, 1.0)

    if config.observation == "ego":
        tensor = egocentric_crop(tensor, x, y, config.ego_radius)

    return tensor


def egocentric_crop(tensor: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    """Crop a window centred on the agent, padding out of bounds with wall.

    An egocentric view makes the policy translation-invariant, which usually
    learns faster, at the cost of hiding anything beyond the window. Whether
    that trade pays off is an ablation, not an assumption -- hence the flag.
    """
    channels = tensor.shape[0]
    padded = np.zeros(
        (channels, tensor.shape[1] + 2 * radius, tensor.shape[2] + 2 * radius),
        dtype=tensor.dtype,
    )
    # Out-of-bounds reads as solid wall, matching the real arena border.
    padded[CH_WALL] = 1.0
    padded[:, radius:-radius, radius:-radius] = tensor

    # Padded index == original index + radius, so a window spanning
    # [x - radius, x + radius] starts at padded index x.
    return np.ascontiguousarray(
        padded[:, x : x + 2 * radius + 1, y : y + 2 * radius + 1]
    )


def observation_shape(config: PPOConfig) -> tuple:
    size = config.spatial_size
    return (N_CHANNELS, size, size)


def action_mask_for(game_state: dict, config: PPOConfig) -> np.ndarray:
    """Safety mask for a state, in real-board action coordinates."""
    from .kit import safety

    field, position, bomb_available, _, _, _, danger, passable = P.game_state_context(
        game_state
    )
    return safety.action_mask(
        field, position, danger, passable, bomb_available, mode=config.safety_mode
    )

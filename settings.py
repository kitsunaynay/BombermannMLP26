import logging
import os
from pathlib import Path

from fallbacks import pygame

# Game properties
# board size (a smaller board may be useful at the beginning)
COLS = 17
ROWS = 17
SCENARIOS = {
    # modes useful for agent development
	"empty": {
        "CRATE_DENSITY": 0, 
        "COIN_COUNT": 0 
    },
    "coin-heaven": {
        "CRATE_DENSITY": 0,
        "COIN_COUNT": 50
    },
    "loot-crate": { 
        "CRATE_DENSITY": 0.75, 
        "COIN_COUNT": 50 
    }, 
    # this is the tournament game mode
    "classic": {
        "CRATE_DENSITY": 0.75,
        "COIN_COUNT": 9
    }
    # Feel free to add more game modes and properties
    # game is created in environment.py -> BombeRLeWorld -> build_arena()
}
MAX_AGENTS = 4

# Round properties
MAX_STEPS = 400

# GUI properties
GRID_SIZE = 30
WIDTH = 1000
HEIGHT = 600
GRID_OFFSET = [(HEIGHT - ROWS * GRID_SIZE) // 2] * 2

ASSET_DIR = Path(__file__).parent / "assets"

AGENT_COLORS = ['blue', 'green', 'yellow', 'pink']

# Game rules
BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2  # = 1 of bomb explosion + N of lingering around

# Rules for agents
TIMEOUT = 0.5
TRAIN_TIMEOUT = float("inf")
REWARD_KILL = 5
REWARD_COIN = 1

# User input
INPUT_MAP = {
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
    pygame.K_RETURN: 'WAIT',
    pygame.K_SPACE: 'BOMB',
}

# Logging levels
#
# Upstream logs every agent at DEBUG into `agent_code/<code>/logs/<agent>.log`.
# That is right for one interactive game and ruinous for anything at volume:
# the six-run stage-4 sweep of 2026-09-08 wrote 2.1 GB across four files (three
# `rule_based_agent` logs at ~600 MB each) for runs whose results were 4 MB, on
# a disk with 5 GB free. The path depends only on the agent's name, so parallel
# runs also share it and overwrite each other.
#
# Default is therefore silence, and logs are opt-in per invocation:
#
#     AOT_LOG_LEVEL=DEBUG python main.py play --agents attackontensor_ql
#
# Accepts any level name (DEBUG, INFO, WARNING, ERROR, CRITICAL). LOG_GAME is
# left at the framework's INFO because `logs/game.log` is one small file per
# run, not one per agent per process.
def _log_level(name: str, default: int) -> int:
    return getattr(logging, os.environ.get(name, "").strip().upper(), default)


_AGENT_LOG_DEFAULT = _log_level("AOT_LOG_LEVEL", logging.CRITICAL)

LOG_GAME = logging.INFO
LOG_AGENT_WRAPPER = _AGENT_LOG_DEFAULT
LOG_AGENT_CODE = _AGENT_LOG_DEFAULT
LOG_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

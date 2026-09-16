"""Official callback API for shielded linear PPO V4."""
from collections import deque
from pathlib import Path

import numpy as np

from .model import LinearPPO

MODEL_PATH = Path(__file__).with_name("model.pkl")


def setup(self):
    self.rng = np.random.default_rng(7301)
    self.own_bomb_positions = set()
    self.recent_positions = deque(maxlen=20)
    self.escape_plan = []
    self.expected_position = None
    self.previous_action = None
    self.current_round = None
    self.shield_fallbacks = 0
    import os
    checkpoint = MODEL_PATH if self.train else Path(os.environ.get("BOMBERMAN_EVAL_MODEL_PATH", str(MODEL_PATH)))
    if not self.train and not checkpoint.exists():
        raise FileNotFoundError(f"Evaluation requires an explicit checkpoint: {checkpoint}")
    self.model = LinearPPO.load(checkpoint) if checkpoint.exists() else LinearPPO()
    self.logger.info("Linear PPO V4 ready at update %d", self.model.n_updates)


def _context(self):
    return {
        "own_bombs": set(self.own_bomb_positions),
        "recent_positions": tuple(self.recent_positions),
        "escape_plan": tuple(self.escape_plan),
        "previous_action": self.previous_action,
    }


def act(self, game_state: dict) -> str:
    position = tuple(game_state["self"][3])
    if self.current_round != game_state["round"]:
        self.current_round = game_state["round"]
        self.own_bomb_positions.clear()
        self.recent_positions.clear()
        self.escape_plan.clear()
        self.expected_position = None
        self.previous_action = None
    if self.expected_position is not None and position != self.expected_position:
        self.escape_plan.clear()  # previous move was blocked or the plan became stale
    live_bombs = {tuple(pos) for pos, _ in game_state.get("bombs", [])}
    self.own_bomb_positions.intersection_update(live_bombs)
    action, record, decision = self.model.choose(
        game_state, self.rng, stochastic=bool(self.train), context=_context(self)
    )
    certificate = decision["certificate"]
    self.escape_plan = list(certificate.path[1:]) if certificate.path else []
    self.expected_position = certificate.path[0] if certificate.path else position
    if decision["fallback"]:
        self.shield_fallbacks += 1
    if self.train:
        self.rollout.append(record)
    if action == "BOMB":
        self.own_bomb_positions.add(position)
        if self.train:
            self.pending_bombs.append({"position": position, "record": record})
    self.recent_positions.append(position)
    self.previous_action = action
    return action

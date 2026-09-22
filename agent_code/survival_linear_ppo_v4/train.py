"""Training callbacks for linear PPO V4."""
from pathlib import Path
import json

import numpy as np
import events as e

from .core import ACTIONS, corridor_depth, earliest_blast_time, shortest_distance, state_features, target_approaches

MODEL_PATH = Path(__file__).with_name("model.pkl")
METRICS_PATH = Path(__file__).with_name("training_metrics.jsonl")

EVENT_REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.SURVIVED_ROUND: 2.0,
    e.INVALID_ACTION: -1.0,
    e.GOT_KILLED: -4.0,
    e.KILLED_SELF: -8.0,
}


def setup_training(self):
    self.rollout = []
    self.pending_bombs = []


def _context(self):
    return {
        "own_bombs": set(self.own_bomb_positions),
        "recent_positions": tuple(self.recent_positions),
        "escape_plan": tuple(self.escape_plan),
        "previous_action": self.previous_action,
    }


def _potential(game_state):
    if game_state is None:
        return 0.0
    field = game_state["field"]
    pos = tuple(game_state["self"][3])
    coins = target_approaches(game_state, game_state.get("coins", []))
    crates = target_approaches(game_state, np.argwhere(field == 1), True)
    opponents = target_approaches(game_state, [other[3] for other in game_state.get("others", [])], True)
    closeness = lambda distance: 1.0 - min(40.0, distance) / 40.0
    target = (
        0.55 * closeness(shortest_distance(game_state, pos, coins))
        + 0.30 * closeness(shortest_distance(game_state, pos, crates))
        + 0.15 * closeness(shortest_distance(game_state, pos, opponents))
    )
    safety = 0.25 * float(earliest_blast_time(game_state, pos) > 1)
    corridor = -0.10 * corridor_depth(field, pos) / 6.0
    return float(target + safety + corridor)


def _transition_reward(self, old_state, new_state, events):
    sparse = sum(EVENT_REWARDS.get(event, 0.0) for event in events)
    shaped = 0.99 * _potential(new_state) - _potential(old_state)
    repeat_penalty = 0.0
    if new_state is not None:
        new_pos = tuple(new_state["self"][3])
        repeat_penalty = -0.03 * min(4, tuple(self.recent_positions).count(new_pos))
    return float(sparse + 0.25 * shaped + repeat_penalty)


def _resolve_bomb(self, events, terminal=False):
    if not self.pending_bombs:
        return
    exploded = e.BOMB_EXPLODED in events
    self_killed = e.KILLED_SELF in events
    if not (exploded or self_killed or terminal):
        return
    pending = self.pending_bombs.pop(0)
    if self_killed:
        bonus = -8.0
    elif exploded:
        crates = events.count(e.CRATE_DESTROYED)
        kills = events.count(e.KILLED_OPPONENT)
        found = events.count(e.COIN_FOUND)
        bonus = 5.0 * kills + 0.25 * crates + 0.10 * found
        bonus += 0.20 if crates + kills else -0.25
    else:
        bonus = 0.0
    record = pending["record"]
    if record["reward"] is None:
        record["delayed_bonus"] = record.get("delayed_bonus", 0.0) + bonus
    else:
        record["reward"] += bonus


def _finish_latest(self, old_state, new_state, events, done):
    if not self.rollout or self.rollout[-1]["reward"] is not None:
        return
    record = self.rollout[-1]
    record["reward"] = _transition_reward(self, old_state, new_state, events) + record.pop("delayed_bonus", 0.0)
    record["done"] = bool(done)
    record["next_value"] = 0.0 if done or new_state is None else float(self.model.value(state_features(new_state, _context(self))))


def _maybe_update(self, force=False):
    if self.pending_bombs:
        return
    minimum = int(self.model.config["min_update_steps"])
    if len(self.rollout) < minimum and not force:
        return
    if not self.rollout:
        return
    metrics = self.model.update(self.rollout, self.rng)
    if metrics:
        with METRICS_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        self.logger.info("PPO V4 update=%d steps=%d KL=%.5f", metrics["update"], metrics["steps"], metrics["kl"])
    self.rollout.clear()


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _resolve_bomb(self, events)
    _finish_latest(self, old_game_state, new_game_state, events, done=False)
    _maybe_update(self)


def end_of_round(self, last_game_state, last_action, events):
    _resolve_bomb(self, events, terminal=True)
    _finish_latest(self, last_game_state, None, events, done=True)
    _maybe_update(self, force=len(self.rollout) >= int(self.model.config["min_update_steps"]))
    self.model.save(MODEL_PATH)

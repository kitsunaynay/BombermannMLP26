"""Shared V4 safety shield and feature engineering; NumPy only at inference."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
MOVE_ACTIONS = ACTIONS[:4]
DELTAS = {
    "UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1),
    "LEFT": (-1, 0), "WAIT": (0, 0), "BOMB": (0, 0),
}
REVERSE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
HORIZON = 8

FEATURE_NAMES = [
    "survival_breadth", "terminal_width", "blast_margin", "survival_duration",
    "danger_improvement", "coin_progress", "crate_progress", "opponent_progress",
    "free_neighbors", "corridor_exit", "dead_end_risk", "opponent_pressure",
    "contested_now", "contested_path", "visit_penalty", "reverse_action",
    "plan_match", "own_blast_exit", "move", "safe_wait", "bomb",
    "bomb_crates", "bomb_opponents", "bomb_has_target", "bomb_escape_speed",
    "bomb_escape_breadth", "future_bomb_value", "trap_opportunity",
]

STATE_FEATURE_NAMES = [
    "bias", "score", "bomb_available", "coins", "crates", "opponents",
    "danger", "coin_distance", "crate_distance", "opponent_distance",
    "free_neighbors", "corridor_depth", "step_fraction", "legal_fraction",
    "best_coin_progress", "best_crate_progress", "best_opponent_progress",
    "best_bomb_value", "escape_mode", "recent_unique_fraction",
]


@dataclass(frozen=True)
class EscapeCertificate:
    safe: bool
    duration: int
    breadth: float
    terminal_width: int
    min_margin: float
    contested_fraction: float
    exit_step: int
    path: tuple


def inside(field, pos):
    x, y = pos
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]


def neighbors(pos):
    x, y = pos
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def blast_tiles(field, origin, power=3):
    """Match items.Bomb.get_blast_coords: only stone walls stop a blast."""
    tiles = [tuple(origin)]
    ox, oy = origin
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for distance in range(1, power + 1):
            pos = (ox + dx * distance, oy + dy * distance)
            if not inside(field, pos) or field[pos] == -1:
                break
            tiles.append(pos)
    return tiles


def _bomb_data(game_state, extra_bomb=None):
    bombs = [(tuple(pos), int(timer)) for pos, timer in game_state.get("bombs", [])]
    if extra_bomb is not None:
        bombs.append((tuple(extra_bomb), 4))
    detonations = {}
    for pos, timer in bombs:
        detonations[pos] = min(detonations.get(pos, 99), max(1, timer + 1))
    return bombs, detonations


def danger_schedule(game_state, extra_bomb=None, horizon=HORIZON):
    field = game_state["field"]
    bombs, detonations = _bomb_data(game_state, extra_bomb)
    horizon = max(horizon, max(detonations.values(), default=1) + 1)
    danger = np.zeros((horizon + 1, *field.shape), dtype=bool)
    explosion = np.asarray(game_state.get("explosion_map", np.zeros_like(field)))
    danger[1][explosion > 0] = True
    blast_lookup = {}
    for pos, _ in bombs:
        blast = tuple(blast_tiles(field, pos))
        blast_lookup[pos] = blast
        detonation = detonations[pos]
        for time_index in (detonation, detonation + 1):
            if time_index <= horizon:
                for tile in blast:
                    danger[(time_index, *tile)] = True
    return danger, detonations, blast_lookup


def earliest_blast_time(game_state, pos):
    field = game_state["field"]
    if np.asarray(game_state.get("explosion_map", np.zeros_like(field)))[tuple(pos)] > 0:
        return 0.0
    earliest = 9.0
    for bomb_pos, timer in game_state.get("bombs", []):
        if tuple(pos) in blast_tiles(field, tuple(bomb_pos)):
            earliest = min(earliest, float(timer))
    return earliest


def static_free_tiles(game_state):
    field = game_state["field"]
    free = field == 0
    return free


def opponent_reachability(game_state, horizon=HORIZON, extra_bomb=None):
    """Over-approximate where any opponent can be at each future step."""
    field = game_state["field"]
    _, detonations = _bomb_data(game_state, extra_bomb)
    current = {tuple(other[3]) for other in game_state.get("others", [])}
    reachable = [set(current)]
    for time_index in range(1, horizon + 1):
        nxt_set = set()
        for pos in current:
            for nxt in (pos, *neighbors(pos)):
                if not inside(field, nxt) or field[nxt] != 0:
                    continue
                if nxt != pos and nxt in detonations and time_index < detonations[nxt]:
                    continue
                nxt_set.add(nxt)
        current = nxt_set
        reachable.append(set(current))
    return reachable


def _free_degree(field, pos):
    return sum(inside(field, nxt) and field[nxt] == 0 for nxt in neighbors(pos))


def corridor_depth(field, start, max_depth=6):
    """Distance to a junction; high values mark deep corridors and pockets."""
    start = tuple(start)
    if not inside(field, start) or field[start] != 0:
        return float(max_depth)
    if _free_degree(field, start) >= 3:
        return 0.0
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        pos, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nxt in neighbors(pos):
            if not inside(field, nxt) or field[nxt] != 0 or nxt in seen:
                continue
            if _free_degree(field, nxt) >= 3:
                return float(depth + 1)
            seen.add(nxt)
            queue.append((nxt, depth + 1))
    return float(max_depth)


def target_approaches(game_state, targets, targets_are_blocked=False):
    field = game_state["field"]
    result = set()
    for target in map(tuple, targets):
        if inside(field, target) and field[target] == 0 and not targets_are_blocked:
            result.add(target)
        else:
            result.update(nxt for nxt in neighbors(target) if inside(field, nxt) and field[nxt] == 0)
    return result


def shortest_distance(game_state, start, targets, max_depth=40):
    targets = set(map(tuple, targets))
    start = tuple(start)
    if not targets:
        return float(max_depth)
    if start in targets:
        return 0.0
    field = game_state["field"]
    blocked = {tuple(pos) for pos, _ in game_state.get("bombs", [])}
    blocked.update(tuple(other[3]) for other in game_state.get("others", []))
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        pos, distance = queue.popleft()
        if distance >= max_depth:
            continue
        for nxt in neighbors(pos):
            if nxt in seen or not inside(field, nxt) or field[nxt] != 0:
                continue
            if nxt in targets:
                return float(distance + 1)
            if nxt in blocked:
                continue
            seen.add(nxt)
            queue.append((nxt, distance + 1))
    return float(max_depth)


def escape_certificate(game_state, start, extra_bomb=None, horizon=HORIZON, risk_averse=False):
    """Time-expanded survival search plus a concrete high-margin path."""
    field = game_state["field"]
    start = tuple(start)
    danger, detonations, blast_lookup = danger_schedule(game_state, extra_bomb, horizon)
    horizon = danger.shape[0] - 1
    opponent_tiles = opponent_reachability(game_state, horizon, extra_bomb)
    new_blast = set(blast_lookup.get(tuple(extra_bomb), ())) if extra_bomb is not None else set()

    if not inside(field, start) or field[start] != 0 or danger[(1, *start)]:
        return EscapeCertificate(False, 0, 0.0, 0, 0.0, 1.0, 99, ())

    initial_contested = start in opponent_tiles[1]
    if risk_averse and initial_contested:
        return EscapeCertificate(False, 0, 0.0, 0, 0.0, 1.0, 99, (start,))

    # pos -> (path, accumulated contested count, minimum blast margin)
    frontier = {start: ((start,), int(initial_contested), 9.0)}
    widths = [1]
    duration = 1
    for time_index in range(1, horizon):
        candidates = {}
        for pos, (path, contested, min_margin) in frontier.items():
            for nxt in (pos, *neighbors(pos)):
                if not inside(field, nxt) or field[nxt] != 0:
                    continue
                # A placed bomb can be waited on, but no bomb can be re-entered.
                if nxt != pos and nxt in detonations and time_index + 1 < detonations[nxt]:
                    continue
                if danger[(time_index + 1, *nxt)]:
                    continue
                is_contested = nxt in opponent_tiles[min(time_index + 1, len(opponent_tiles) - 1)]
                # Near-term collisions are treated adversarially only in escape mode.
                if risk_averse and time_index + 1 <= 2 and is_contested:
                    continue
                margins = []
                for bomb_pos, detonation in detonations.items():
                    if nxt in blast_lookup[bomb_pos] and detonation >= time_index + 1:
                        margins.append(detonation - (time_index + 1))
                local_margin = float(min(margins, default=9))
                candidate = (
                    path + (nxt,),
                    contested + int(is_contested),
                    min(min_margin, local_margin),
                )
                old = candidates.get(nxt)
                candidate_exit = next((i for i, tile in enumerate(candidate[0], start=1) if tile not in new_blast), 99) if new_blast else 0
                old_exit = next((i for i, tile in enumerate(old[0], start=1) if tile not in new_blast), 99) if old and new_blast else 0
                rank = (candidate_exit, candidate[1], -candidate[2], corridor_depth(field, nxt))
                old_rank = (old_exit, old[1], -old[2], corridor_depth(field, nxt)) if old else None
                if old is None or rank < old_rank:
                    candidates[nxt] = candidate
        frontier = candidates
        if not frontier:
            break
        duration = time_index + 1
        widths.append(len(frontier))

    if frontier:
        best_pos, best_data = min(
            frontier.items(),
            key=lambda item: (
                next((i for i, tile in enumerate(item[1][0], start=1) if tile not in new_blast), 99) if new_blast else 0,
                item[1][1], corridor_depth(field, item[0]), -item[1][2], -_free_degree(field, item[0]),
            ),
        )
        path, contested, min_margin = best_data
    else:
        path, contested, min_margin = (), horizon, 0.0

    exit_step = 0
    if new_blast and path:
        exit_step = next((index for index, pos in enumerate(path, start=1) if pos not in new_blast), 99)
    breadth = float(np.mean([min(1.0, width / 8.0) for width in widths]))
    safe = duration >= horizon
    return EscapeCertificate(
        safe=safe,
        duration=duration,
        breadth=breadth,
        terminal_width=len(frontier),
        min_margin=float(min_margin),
        contested_fraction=float(contested / max(1, len(path))),
        exit_step=int(exit_step),
        path=tuple(path),
    )


def bomb_targets(game_state, origin):
    field = game_state["field"]
    blast = set(blast_tiles(field, tuple(origin)))
    crates = {tuple(pos) for pos in np.argwhere(field == 1)}
    opponents = {tuple(other[3]) for other in game_state.get("others", [])}
    return sum(tile in crates for tile in blast), sum(tile in opponents for tile in blast)


def _physical_actions(game_state):
    field = game_state["field"]
    _, _, bomb_available, pos = game_state["self"]
    pos = tuple(pos)
    occupied = {tuple(p) for p, _ in game_state.get("bombs", [])}
    occupied.update(tuple(other[3]) for other in game_state.get("others", []))
    physical = np.zeros(len(ACTIONS), dtype=bool)
    for index, action in enumerate(ACTIONS):
        dx, dy = DELTAS[action]
        nxt = (pos[0] + dx, pos[1] + dy)
        if action in MOVE_ACTIONS:
            physical[index] = inside(field, nxt) and field[nxt] == 0 and nxt not in occupied
        elif action == "WAIT":
            physical[index] = True
        else:
            physical[index] = bool(bomb_available) and pos not in occupied
    return physical


def safe_action_data(game_state, context=None):
    """Return action features, shield mask, and per-action certificates."""
    context = context or {}
    shield_mode = context.get("shield_mode", "full")
    if shield_mode not in {"full", "basic"}:
        raise ValueError(f"Unknown shield mode: {shield_mode}")
    field = game_state["field"]
    _, _, bomb_available, pos = game_state["self"]
    pos = tuple(pos)
    coins = [tuple(item) for item in game_state.get("coins", [])]
    crates = [tuple(item) for item in np.argwhere(field == 1)]
    opponents = [tuple(other[3]) for other in game_state.get("others", [])]
    coin_targets = target_approaches(game_state, coins)
    crate_targets = target_approaches(game_state, crates, targets_are_blocked=True)
    opponent_targets = target_approaches(game_state, opponents, targets_are_blocked=True)
    current_distances = (
        shortest_distance(game_state, pos, coin_targets),
        shortest_distance(game_state, pos, crate_targets),
        shortest_distance(game_state, pos, opponent_targets),
    )
    current_danger = earliest_blast_time(game_state, pos)
    current_corridor = corridor_depth(field, pos)
    own_bombs = {tuple(item) for item in context.get("own_bombs", set())}
    own_blast = set()
    for own_bomb in own_bombs:
        own_blast.update(blast_tiles(field, own_bomb))
    escape_mode = bool(own_bombs) or current_danger <= 3
    recent = [tuple(item) for item in context.get("recent_positions", ())]
    previous_action = context.get("previous_action")
    plan = [tuple(item) for item in context.get("escape_plan", ())]
    planned_next = plan[0] if plan else None
    physical = _physical_actions(game_state)
    phi = np.zeros((len(ACTIONS), len(FEATURE_NAMES)), dtype=np.float64)
    legal = np.zeros(len(ACTIONS), dtype=bool)
    details = []

    for index, action in enumerate(ACTIONS):
        dx, dy = DELTAS[action]
        nxt = (pos[0] + dx, pos[1] + dy)
        extra_bomb = pos if action == "BOMB" and physical[index] else None
        cert = escape_certificate(
            game_state, nxt, extra_bomb=extra_bomb,
            risk_averse=escape_mode or action == "BOMB",
        ) if physical[index] else EscapeCertificate(False, 0, 0.0, 0, 0.0, 1.0, 99, ())
        crate_hits, opponent_hits = bomb_targets(game_state, pos) if action == "BOMB" else (0, 0)
        has_target = crate_hits + opponent_hits > 0
        robust_bomb = (
            action != "BOMB" or (
                bool(bomb_available) and has_target and cert.safe
                and 1 <= cert.exit_step <= 4
                and cert.terminal_width >= 2
                and cert.breadth >= 0.20
            )
        )
        legal[index] = bool(physical[index] and cert.safe and robust_bomb)

        next_distances = (
            shortest_distance(game_state, nxt, coin_targets) if physical[index] else current_distances[0],
            shortest_distance(game_state, nxt, crate_targets) if physical[index] else current_distances[1],
            shortest_distance(game_state, nxt, opponent_targets) if physical[index] else current_distances[2],
        )
        progress = [np.tanh((old - new) / 3.0) for old, new in zip(current_distances, next_distances)]
        next_danger = earliest_blast_time(game_state, nxt) if physical[index] else 0.0
        next_corridor = corridor_depth(field, nxt) if physical[index] else 6.0
        nearest_opponent = min((abs(nxt[0] - ox) + abs(nxt[1] - oy) for ox, oy in opponents), default=20)
        pressure = math.exp(-nearest_opponent / 3.0)
        free_neighbors = sum(
            inside(field, candidate) and field[candidate] == 0 for candidate in neighbors(nxt)
        ) / 4.0 if inside(field, nxt) else 0.0
        future_crates = future_opponents = 0
        if physical[index] and action in MOVE_ACTIONS and bomb_available:
            future_crates, future_opponents = bomb_targets(game_state, nxt)
        future_bomb_value = min(1.0, (future_crates + 3 * future_opponents) / 5.0)
        trap_opportunity = float(opponent_hits > 0) * min(1.0, next_corridor / 4.0)
        current_in_own_blast = pos in own_blast
        next_in_own_blast = nxt in own_blast

        phi[index] = [
            cert.breadth,
            min(1.0, cert.terminal_width / 8.0),
            min(1.0, cert.min_margin / 5.0),
            min(1.0, cert.duration / HORIZON),
            np.clip((next_danger - current_danger) / 4.0, -1.0, 1.0),
            progress[0], progress[1], progress[2], free_neighbors,
            np.clip((current_corridor - next_corridor) / 6.0, -1.0, 1.0),
            min(1.0, next_corridor / 6.0) * pressure,
            pressure,
            float(cert.contested_fraction > 0 and len(cert.path) > 0 and cert.path[0] in opponent_reachability(game_state, 1)[1]),
            cert.contested_fraction,
            min(1.0, recent[-16:].count(nxt) / 4.0),
            float(action == REVERSE.get(previous_action)),
            float(planned_next is not None and nxt == planned_next),
            float(current_in_own_blast) - float(next_in_own_blast),
            float(action in MOVE_ACTIONS),
            float(action == "WAIT" and cert.safe),
            float(action == "BOMB"),
            min(1.0, crate_hits / 4.0),
            min(1.0, opponent_hits / 2.0),
            float(has_target),
            0.0 if action != "BOMB" or cert.exit_step >= 99 else (5.0 - cert.exit_step) / 4.0,
            cert.breadth if action == "BOMB" else 0.0,
            future_bomb_value,
            trap_opportunity,
        ]
        details.append(cert)

    if shield_mode == "basic":
        # Same phi/details as full mode; no future certificate gates or exit forcing.
        immediate, _, _ = danger_schedule(game_state)
        basic = physical.copy()
        for i, action in enumerate(ACTIONS):
            dx, dy = DELTAS[action]
            nxt = (pos[0] + dx, pos[1] + dy)
            if physical[i]:
                basic[i] = not immediate[(1, *nxt)]
        fallback = not basic.any()
        if fallback:
            basic = physical.copy()  # No certificate-based least-risk ranking.
        return phi, basic, {"certificates": details, "fallback": fallback,
                            "escape_mode": escape_mode, "shield_mode": "basic"}

    # Commit to leaving an own blast when a certified immediate exit exists.
    if own_bombs and legal.any() and pos in own_blast:
        exits = legal & (phi[:, FEATURE_NAMES.index("own_blast_exit")] > 0)
        if exits.any():
            legal = exits

    fallback = False
    if not legal.any():
        # Maximize time alive, then minimize opponent conflict, then maximize route width.
        fallback = True
        candidates = np.flatnonzero(physical)
        if len(candidates) == 0:
            candidates = np.array([ACTIONS.index("WAIT")])
        ranks = [
            (details[i].duration, -details[i].contested_fraction, details[i].terminal_width, details[i].breadth)
            for i in candidates
        ]
        best_rank = max(ranks)
        for i, rank in zip(candidates, ranks):
            if rank == best_rank:
                legal[i] = True

    return phi, legal, {"certificates": details, "fallback": fallback, "escape_mode": escape_mode}


def state_features(game_state, context=None, action_data=None):
    if game_state is None:
        return np.zeros(len(STATE_FEATURE_NAMES), dtype=np.float64)
    context = context or {}
    field = game_state["field"]
    _, score, bomb_available, pos = game_state["self"]
    pos = tuple(pos)
    coins = [tuple(item) for item in game_state.get("coins", [])]
    crates = [tuple(item) for item in np.argwhere(field == 1)]
    opponents = [tuple(other[3]) for other in game_state.get("others", [])]
    coin_targets = target_approaches(game_state, coins)
    crate_targets = target_approaches(game_state, crates, True)
    opponent_targets = target_approaches(game_state, opponents, True)
    if action_data is None:
        phi, legal, metadata = safe_action_data(game_state, context)
    else:
        phi, legal, metadata = action_data
    recent = [tuple(item) for item in context.get("recent_positions", ())]
    unique_fraction = len(set(recent[-16:])) / max(1, len(recent[-16:]))
    return np.asarray([
        1.0,
        np.tanh(float(score) / 10.0),
        float(bool(bomb_available)),
        min(1.0, len(coins) / 20.0),
        min(1.0, len(crates) / 100.0),
        len(opponents) / 3.0,
        np.clip((4.0 - earliest_blast_time(game_state, pos)) / 4.0, 0.0, 1.0),
        np.tanh(shortest_distance(game_state, pos, coin_targets) / 10.0),
        np.tanh(shortest_distance(game_state, pos, crate_targets) / 10.0),
        np.tanh(shortest_distance(game_state, pos, opponent_targets) / 10.0),
        _free_degree(field, pos) / 4.0,
        corridor_depth(field, pos) / 6.0,
        min(1.0, game_state.get("step", 0) / 400.0),
        float(legal.mean()),
        float(phi[legal, FEATURE_NAMES.index("coin_progress")].max(initial=0.0)),
        float(phi[legal, FEATURE_NAMES.index("crate_progress")].max(initial=0.0)),
        float(phi[legal, FEATURE_NAMES.index("opponent_progress")].max(initial=0.0)),
        float((phi[legal, FEATURE_NAMES.index("bomb_crates")] + 3 * phi[legal, FEATURE_NAMES.index("bomb_opponents")]).max(initial=0.0)),
        float(metadata["escape_mode"]),
        unique_fraction,
    ], dtype=np.float64)


def action_matrix(game_state, context=None):
    action_data = safe_action_data(game_state, context)
    phi, legal, metadata = action_data
    state = state_features(game_state, context, action_data)
    matrix = np.concatenate([phi, np.repeat(state[None, :], len(ACTIONS), axis=0)], axis=1)
    return matrix, legal, metadata

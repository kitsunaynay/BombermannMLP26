"""Tests for the Q-learning agent."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

import settings as s
from agent_code.attackontensor_ql import features as F
from agent_code.attackontensor_ql.config import VARIANTS, QLConfig
from agent_code.attackontensor_ql.kit import symmetry
from agent_code.attackontensor_ql.kit.actions import ACTIONS, BOMB, N_ACTIONS, WAIT
from agent_code.attackontensor_ql.qtable import QTable
from helpers import build_arena


def make_game_state(field=None, position=(1, 1), coins=(), bombs=(), others=(), step=1):
    field = build_arena(0) if field is None else field
    return {
        "round": 1,
        "step": step,
        "field": field,
        "self": ("me", 0, True, position),
        "others": [("them", 0, True, xy) for xy in others],
        "bombs": list(bombs),
        "coins": list(coins),
        "explosion_map": np.zeros(field.shape),
        "user_input": None,
    }


def rotate_game_state(game_state, transform):
    """Re-express an entire game state on a D4-transformed board."""
    field = game_state["field"]
    size = field.shape[0]
    assert field.shape[0] == field.shape[1], "rotation helper assumes a square board"

    def move(xy):
        marker = np.zeros_like(field)
        marker[xy] = 1
        moved = symmetry.transform_grid(marker, transform)
        (nx,), (ny,) = np.nonzero(moved)
        return int(nx), int(ny)

    name, score, bomb, position = game_state["self"]
    return {
        **game_state,
        "field": symmetry.transform_grid(field, transform),
        "self": (name, score, bomb, move(position)),
        "others": [(n, sc, b, move(xy)) for (n, sc, b, xy) in game_state["others"]],
        "bombs": [(move(xy), t) for xy, t in game_state["bombs"]],
        "coins": [move(xy) for xy in game_state["coins"]],
        "explosion_map": symmetry.transform_grid(game_state["explosion_map"], transform),
    }


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_features_have_the_declared_shape_and_ranges(variant):
    config = QLConfig(variant=variant)
    state = make_game_state(coins=[(3, 1)])

    key = F.state_to_features(state, config)

    assert len(key) == len(config.feature_names)
    assert all(isinstance(v, int) for v in key)
    for value, cardinality in zip(key, F.feature_cardinalities(config)):
        assert 0 <= value < cardinality


def test_features_are_none_for_a_missing_state():
    """Dead agents are handed None (environment.py:397)."""
    assert F.state_to_features(None, QLConfig()) is None
    assert F.extract(None, QLConfig()) is None


def test_features_are_deterministic():
    config = QLConfig()
    state = make_game_state(coins=[(3, 1), (5, 5)], bombs=[((1, 3), 2)])

    keys = {F.state_to_features(state, config) for _ in range(10)}

    assert len(keys) == 1, "feature extraction must not depend on random state"


def test_coin_direction_points_at_the_coin():
    config = QLConfig(variant="full", use_symmetry=False)
    field = np.full((9, 9), -1, dtype=int)
    field[1:8, 1] = 0

    values = F.compute_all_features(make_game_state(field, (3, 1), coins=[(6, 1)]))

    assert values["coin_dir"] == 2, "coin lies to the RIGHT"


def test_escape_if_bomb_is_false_in_a_sealed_pocket():
    field = np.full((9, 9), -1, dtype=int)
    field[1:4, 1] = 0  # three tiles: the whole corridor is inside our own blast

    values = F.compute_all_features(make_game_state(field, (1, 1)))

    assert values["escape_if_bomb"] == 0


def test_escape_if_bomb_is_true_with_room_to_run():
    field = np.full((12, 9), -1, dtype=int)
    field[1:9, 1] = 0

    values = F.compute_all_features(make_game_state(field, (1, 1)))

    assert values["escape_if_bomb"] == 1


def _pocket_field():
    """A corridor at y=1 with a one-tile dead-end branch at (3, 2)."""
    field = np.full((9, 9), -1, dtype=int)
    field[1:8, 1] = 0
    field[3, 2] = 0
    return field


def test_in_dead_end_is_true_only_on_the_pocket_tile():
    field = _pocket_field()

    assert F.compute_all_features(make_game_state(field, (3, 2)))["in_dead_end"] == 1
    assert F.compute_all_features(make_game_state(field, (3, 1)))["in_dead_end"] == 0


def test_opponent_in_dead_end_detects_a_nearby_trapped_opponent():
    field = _pocket_field()

    trapped = F.compute_all_features(make_game_state(field, (5, 1), others=[(3, 2)]))
    assert trapped["opponent_in_dead_end"] == 1

    free = F.compute_all_features(make_game_state(field, (5, 1), others=[(4, 1)]))
    assert free["opponent_in_dead_end"] == 0


def test_opponent_in_dead_end_ignores_a_far_away_opponent():
    field = _pocket_field()

    values = F.compute_all_features(make_game_state(field, (7, 1), others=[(3, 2)]))

    assert values["opponent_in_dead_end"] == 0


def test_can_seal_opponent_when_standing_on_its_only_exit():
    field = _pocket_field()

    on_exit = F.compute_all_features(make_game_state(field, (3, 1), others=[(3, 2)]))
    assert on_exit["can_seal_opponent"] == 1

    adjacent_to_exit = F.compute_all_features(make_game_state(field, (4, 1), others=[(3, 2)]))
    assert adjacent_to_exit["can_seal_opponent"] == 1

    far_from_exit = F.compute_all_features(make_game_state(field, (5, 1), others=[(3, 2)]))
    assert far_from_exit["opponent_in_dead_end"] == 1, "still within range 3 of the trapped opponent"
    assert far_from_exit["can_seal_opponent"] == 0, "two tiles from the exit, not on or adjacent to it"


def test_state_space_size_is_reported():
    assert F.state_space_size(QLConfig(variant="full")) > F.state_space_size(
        QLConfig(variant="compact")
    )


# --------------------------------------------------------------------------
# Symmetry round-trip -- the highest-value invariant in this file
# --------------------------------------------------------------------------


def open_arena(size: int = 11) -> np.ndarray:
    """Walled box with an open interior and no crates, so targets are unique."""
    field = np.full((size, size), -1, dtype=int)
    field[1 : size - 1, 1 : size - 1] = 0
    return field


@pytest.mark.parametrize("transform", symmetry.TRANSFORMS)
def test_features_are_equivariant_on_a_tie_free_board(transform):
    """Relabelling features must equal recomputing them on the rotated board.

    This is the property the whole canonicalisation rests on. It is asserted on
    a board with a single unique nearest target, because equidistant targets
    force a tie-break and no deterministic rule breaks a symmetric tie
    symmetrically (see the note in features.py).
    """
    state = make_game_state(open_arena(), (5, 5), coins=[(8, 5)])

    relabelled = F._transform_values(F.compute_all_features(state), transform)
    recomputed = F.compute_all_features(rotate_game_state(state, transform))

    assert relabelled == recomputed


@pytest.mark.parametrize("transform", symmetry.TRANSFORMS)
def test_canonical_key_is_invariant_under_board_rotation(transform):
    """A rotated board must canonicalise to the same key.

    This is what makes the 8x state-space reduction sound: experience learnt in
    one corner has to be reusable in the other three.
    """
    config = QLConfig(variant="full", use_symmetry=True)
    state = make_game_state(open_arena(), (5, 5), coins=[(8, 5)])

    original = F.extract(state, config)
    rotated = F.extract(rotate_game_state(state, transform), config)

    assert original.key == rotated.key


@pytest.mark.parametrize("transform", symmetry.TRANSFORMS)
def test_action_round_trip_through_the_canonical_frame(transform):
    for action in range(N_ACTIONS):
        canonical = symmetry.transform_action(action, transform)
        assert symmetry.inverse_transform_action(canonical, transform) == action


def test_canonicalisation_collapses_a_tie_free_orbit_completely():
    state = make_game_state(open_arena(), (5, 5), coins=[(8, 5)])
    views = [rotate_game_state(state, t) for t in symmetry.TRANSFORMS]

    with_symmetry = {F.extract(v, QLConfig(use_symmetry=True)).key for v in views}
    without = {F.extract(v, QLConfig(use_symmetry=False)).key for v in views}

    assert len(with_symmetry) == 1
    assert len(without) > 1, "the orbit really is 8 distinct raw states"


def test_canonicalisation_still_helps_on_a_cluttered_board():
    """On boards full of equidistant crates, collapse is partial but real.

    Quantifying the shortfall rather than asserting perfection keeps the
    reported state-space reduction honest.
    """
    config_on = QLConfig(variant="full", use_symmetry=True)
    config_off = QLConfig(variant="full", use_symmetry=False)
    state = make_game_state(coins=[(3, 1)], bombs=[((5, 5), 2)], others=[(7, 7)])
    views = [rotate_game_state(state, t) for t in symmetry.TRANSFORMS]

    with_symmetry = {F.extract(v, config_on).key for v in views}
    without = {F.extract(v, config_off).key for v in views}

    assert len(with_symmetry) < len(without)


# --------------------------------------------------------------------------
# Q-table
# --------------------------------------------------------------------------


def test_qtable_learns_toward_the_target():
    table = QTable(double=False, seed=0)
    state = (0, 0, 0)

    for _ in range(200):
        table.learn(state, 1, 1.0, None, 0.0, 0.3)

    assert table.q(state)[1] == pytest.approx(1.0, abs=1e-3)


def test_qtable_greedy_respects_the_mask():
    table = QTable(double=False, seed=0)
    state = (1,)
    table.q(state)[:] = [5.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[1] = True

    assert table.greedy(state, mask) == 1, "must not pick the masked-out best action"


def test_qtable_greedy_ignores_an_all_false_mask():
    """When death is certain the mask can be empty; acting must still work."""
    table = QTable(double=False, seed=0)
    state = (2,)
    table.q(state)[:] = [0.0, 9.0, 0.0, 0.0, 0.0, 0.0]

    assert table.greedy(state, np.zeros(N_ACTIONS, dtype=bool)) == 1


def test_qtable_ties_are_broken_randomly():
    """A fresh all-zero table must not deterministically pick action 0."""
    table = QTable(double=False, seed=0)
    picks = {table.greedy((3,)) for _ in range(100)}
    assert len(picks) > 1


def test_double_q_uses_both_tables():
    table = QTable(double=True, seed=0)
    state = (4,)

    for _ in range(500):
        table.learn(state, 2, 1.0, None, 0.0, 0.2)

    assert table.q(state)[2] == pytest.approx(1.0, abs=1e-2)


def test_qtable_round_trips_through_disk(tmp_path):
    table = QTable(double=True, seed=1)
    table.learn((1, 2, 3), 0, 2.5, None, 0.0, 0.5)
    path = tmp_path / "q.pkl"

    table.save(path)
    restored = QTable.load(path)

    assert restored.n_states == table.n_states
    assert np.allclose(restored.q((1, 2, 3)), table.q((1, 2, 3)))
    assert restored.double == table.double


def test_qtable_rejects_a_foreign_format(tmp_path):
    import pickle

    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps({"version": 999}))

    with pytest.raises(ValueError, match="format version"):
        QTable.load(path)


def test_qtable_diagnostics():
    table = QTable(double=False, seed=0)
    table.learn((0,), 0, 10.0, None, 0.0, 1.0)

    assert table.n_states == 1
    assert table.max_abs_q == pytest.approx(10.0)
    assert table.is_finite()
    assert 0.0 < table.coverage() <= 1.0


# --------------------------------------------------------------------------
# Representation binding
# --------------------------------------------------------------------------


def test_metadata_survives_a_checkpoint_round_trip(tmp_path):
    table = QTable(double=False, seed=0, metadata={"variant": "compact", "use_symmetry": True})
    table.learn((1, 2, 3), 0, 1.0, None, 0.0, 0.5)
    path = tmp_path / "q.pkl"

    table.save(path)
    restored = QTable.load(path)

    assert restored.metadata["variant"] == "compact"
    assert restored.metadata["use_symmetry"] is True


def test_key_length_reports_the_feature_count():
    table = QTable(double=False, seed=0)
    assert table.key_length() is None, "empty table has no key length yet"

    table.learn((1, 2, 3, 4), 0, 1.0, None, 0.0, 0.5)
    assert table.key_length() == 4


def test_evaluation_adopts_the_checkpoints_variant(tmp_path, monkeypatch):
    """A compact-trained table must not be queried with the full variant.

    The tournament sets no environment variables, so the config falls back to
    its default (`full`). Without adopting the checkpoint's representation every
    lookup would miss and the agent would play as if untrained -- with no error.
    """
    from agent_code.attackontensor_ql import callbacks

    # Train-side: a table stamped as compact.
    table = QTable(double=True, seed=0)
    table.metadata.update(variant="compact", use_symmetry=True)
    model = tmp_path / "q_table.pkl"
    table.save(model)

    monkeypatch.setenv("AOT_QL_MODEL_FILE", str(model))
    monkeypatch.setenv("AOT_QL_VARIANT", "full")  # deliberately wrong

    agent = SimpleNamespace(train=False, logger=logging.getLogger("test.adopt"))
    agent.logger.handlers = [logging.NullHandler()]
    callbacks.setup(agent)

    assert agent.config.variant == "compact", "checkpoint must win over the config"
    assert len(agent.config.feature_names) == len(VARIANTS["compact"])


def test_trained_table_keys_match_the_configured_variant(tmp_path, monkeypatch):
    """End-to-end: train under compact, evaluate, and confirm lookups hit."""
    from agent_code.attackontensor_ql import callbacks, train

    monkeypatch.setenv("AOT_QL_MODEL_FILE", str(tmp_path / "q_table.pkl"))
    monkeypatch.setenv("AOT_QL_METRICS_FILE", str(tmp_path / "m.csv"))
    monkeypatch.setenv("AOT_QL_VARIANT", "compact")

    trainer = SimpleNamespace(train=True, logger=logging.getLogger("test.train"))
    trainer.logger.handlers = [logging.NullHandler()]
    callbacks.setup(trainer)
    train.setup_training(trainer)

    old, new = make_game_state(step=1), make_game_state(step=2, position=(2, 1))
    train.game_events_occurred(trainer, old, "RIGHT", new, ["MOVED_RIGHT"])
    train.end_of_round(trainer, old, "RIGHT", ["MOVED_RIGHT", "SURVIVED_ROUND"])
    trainer.q.save(trainer.config.model_path)

    assert trainer.q.key_length() == len(VARIANTS["compact"])

    # Now load it the way the tournament would: no variant set at all.
    monkeypatch.delenv("AOT_QL_VARIANT")
    player = SimpleNamespace(train=False, logger=logging.getLogger("test.play"))
    player.logger.handlers = [logging.NullHandler()]
    callbacks.setup(player)

    assert player.config.variant == "compact"
    view = F.extract(make_game_state(step=1), player.config)
    assert len(view.key) == player.q.key_length(), "keys must be lookup-compatible"


def test_key_lengths_reports_every_shape():
    """A resumed run can leave two key shapes in one table.

    Sampling a single key (as an earlier version did) reports a total mismatch
    when most lookups are fine. Measured on the real Task-2 artifact: 130 dead
    9-feature keys alongside 510 live 12-feature ones, and pruning the dead ones
    left the benchmark score bit-identical.
    """
    table = QTable(double=False, seed=0)
    for i in range(3):
        table.learn((i,) * 9, 0, 1.0, None, 0.0, 0.5)
    for i in range(7):
        table.learn((i,) * 12, 0, 1.0, None, 0.0, 0.5)

    assert table.key_lengths() == {9: 3, 12: 7}
    assert table.key_length() == 12, "reports the majority shape, not the first seen"


def test_prune_foreign_keys_removes_only_unreachable_entries():
    table = QTable(double=False, seed=0)
    table.learn((1,) * 9, 0, 5.0, None, 0.0, 1.0)
    table.learn((2,) * 12, 0, 7.0, None, 0.0, 1.0)
    keep = table.q((2,) * 12).copy()

    removed = table.prune_foreign_keys(12)

    assert removed >= 1
    assert table.key_lengths() == {12: 1}
    assert np.allclose(table.q((2,) * 12), keep), "surviving values must be untouched"


def test_mixed_key_shapes_warn_rather_than_error(tmp_path, monkeypatch, caplog):
    """Unreachable leftovers are dead weight, not a failure."""
    import logging as _logging

    from agent_code.attackontensor_ql import callbacks

    table = QTable(double=True, seed=0)
    table.metadata.update(variant="full", use_symmetry=True)
    for i in range(2):
        table.learn((i,) * 9, 0, 1.0, None, 0.0, 0.5)   # stale shape
    for i in range(5):
        table.learn((i,) * 12, 0, 1.0, None, 0.0, 0.5)  # current shape
    model = tmp_path / "q_table.pkl"
    table.save(model)

    monkeypatch.setenv("AOT_QL_MODEL_FILE", str(model))
    agent = SimpleNamespace(train=False, logger=_logging.getLogger("test.mixed"))
    agent.logger.handlers = [_logging.NullHandler()]

    with caplog.at_level(_logging.WARNING, logger="test.mixed"):
        callbacks.setup(agent)

    messages = " ".join(record.message for record in caplog.records)
    assert "dead weight" in messages
    assert not any(r.levelno >= _logging.ERROR for r in caplog.records), (
        "a partially usable table must not be reported as a total failure"
    )


def test_safety_mode_travels_with_the_checkpoint(tmp_path, monkeypatch):
    """A table must be replayed under the mask it was trained behind.

    ``safety_mode`` is not part of the feature key, so it was not stamped into
    the table and not adopted at load. But it is part of the trained policy: a
    table trained behind ``hard`` never observes a no-escape bomb drop, because
    the mask always removed it, so it never learns to avoid one. Replayed under
    the default ``soft`` the same table went from 38.7 coins and 0% suicide to
    1.35 coins and 100% suicide, and the ablation looked like the safety filter
    causing suicide rather than preventing it.
    """
    from agent_code.attackontensor_ql import callbacks
    from agent_code.attackontensor_ql.qtable import QTable

    path = tmp_path / "q_table.pkl"
    table = QTable(double=False, seed=0)
    table.metadata.update(variant="full", use_symmetry=True, safety_mode="hard")
    table.learn((0,) * 12, 0, 1.0, None, 0.0, 1.0)
    table.save(path)

    monkeypatch.setenv("AOT_QL_MODEL_FILE", str(path))
    monkeypatch.setenv("AOT_QL_SAFETY_MODE", "soft")   # config disagrees

    agent = SimpleNamespace(train=False, logger=logging.getLogger("test.safety"))
    callbacks.setup(agent)

    assert agent.config.safety_mode == "hard", (
        "the checkpoint's safety_mode must win over the configured one"
    )


def test_seed_makes_the_agents_rng_reproducible(tmp_path, monkeypatch):
    """``AOT_QL_SEED`` has to reach the table, not just ``self.rng``.

    Every stochastic decision the agent makes -- the epsilon-greedy draw, the
    greedy tie-break and the Double-Q coin flip -- comes from ``QTable._rng``,
    which was constructed unseeded at all four call sites. Seeding only
    ``self.rng`` in ``setup`` would look like it worked and change nothing, so
    this pins the table's own stream.
    """
    from agent_code.attackontensor_ql import callbacks

    def draws(seed):
        monkeypatch.setenv("AOT_QL_SEED", str(seed))
        monkeypatch.setenv("AOT_QL_MODEL_FILE", str(tmp_path / "missing.pkl"))
        agent = SimpleNamespace(train=False, logger=logging.getLogger("test.seed"))
        callbacks.setup(agent)
        # An empty table ties on every action, so greedy() is pure tie-break.
        return [agent.q.greedy((0,) * 12) for _ in range(24)]

    assert draws(11) == draws(11)
    assert draws(11) != draws(12), "different seeds must give different streams"


def test_the_tournament_default_leaves_the_agent_unseeded():
    """The default must stay unseeded: a fixed seed in the tournament would make
    the agent play the identical tie-break sequence in every game.
    """
    assert QLConfig().seed == -1
    assert QLConfig().rng_seed is None


def test_evaluation_is_greedy_by_default(tmp_path, monkeypatch):
    """`eval_epsilon` must default to 0.

    Evaluation noise was measured as a possible escape from greedy limit cycles
    and rejected: on collapsed Task-2 checkpoints it roughly doubled coins
    (2.75 -> 6.60) but drove survival from 100% to 85% at eps = 0.02 and to 20%
    at eps = 0.10, because a random action beside a live bomb is fatal. The knob
    stays for reproducing that measurement; the default must not change.
    """
    assert QLConfig().eval_epsilon == 0.0

    monkeypatch.setenv("AOT_QL_EVAL_EPSILON", "0.05")
    assert QLConfig.load().eval_epsilon == pytest.approx(0.05)

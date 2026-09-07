"""The double-delivery contract, tested against the framework's exact pattern.

``do_step`` (environment.py:158) calls ``send_game_events`` and then, if the
round is over, ``end_round``:

* a **surviving** agent receives its final transition through
  ``game_events_occurred`` *and again* through ``end_of_round``, with the same
  ``last_game_state`` and ``last_action``, and ``SURVIVED_ROUND`` appended to the
  same list object in between;
* an agent that **died** receives it only through ``end_of_round``, because
  ``send_game_events`` skips the dead (environment.py:469).

Naively appending experience in both callbacks double-counts every surviving
episode's last transition. Nothing about a training curve makes that visible,
which is exactly why it is pinned here.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

import events as e
import settings as s
from helpers import build_arena


def make_self(train: bool = True) -> SimpleNamespace:
    logger = logging.getLogger("test.agent")
    logger.handlers = [logging.NullHandler()]
    logger.setLevel(logging.CRITICAL)
    return SimpleNamespace(train=train, logger=logger)


def make_state(step: int, position=(1, 1), coins=(), bombs=()) -> dict:
    field = build_arena(0)
    return {
        "round": 1,
        "step": step,
        "field": field,
        "self": ("me", 0, True, position),
        "others": [],
        "bombs": list(bombs),
        "coins": list(coins),
        "explosion_map": np.zeros(field.shape),
        "user_input": None,
    }


# --------------------------------------------------------------------------
# Q-learning
# --------------------------------------------------------------------------


@pytest.fixture
def ql_agent(tmp_path, monkeypatch):
    from agent_code.attackontensor_ql import callbacks, train

    # Point the agent at a tmp checkpoint *before* setup runs. Otherwise setup
    # loads the repository's real q_table.pkl and the test starts with whatever
    # visit counts that artifact carries -- and its atexit save would overwrite
    # the trained model. model_path is AGENT_DIR / model_file, and pathlib
    # returns the right-hand side when it is absolute.
    monkeypatch.setenv("AOT_QL_MODEL_FILE", str(tmp_path / "q_table.pkl"))
    monkeypatch.setenv("AOT_QL_METRICS_FILE", str(tmp_path / "metrics.csv"))

    agent = make_self(train=True)
    callbacks.setup(agent)
    train.setup_training(agent)

    assert total_updates(agent.q) == 0, "fixture must start from an empty table"
    return agent, train


def total_updates(table) -> int:
    """Number of learn() calls, from the visit counters."""
    return sum(int(counts.sum()) for counts in table._visits.values())


def last_metrics_row(agent) -> dict:
    """The row written for the round that just ended.

    Round statistics are reset by ``_finish_round`` once the row is written, so
    the CSV -- not the live namespace -- is where a finished round's numbers are.
    """
    import csv

    with open(agent._metrics_path) as stream:
        rows = list(csv.DictReader(stream))
    assert rows, "no metrics row was written"
    return rows[-1]


def test_ql_survivor_final_transition_is_counted_once(ql_agent):
    agent, train = ql_agent
    old, new = make_state(1), make_state(2, position=(2, 1))

    # The framework hands the SAME list object to both callbacks.
    events = [e.MOVED_RIGHT]
    train.game_events_occurred(agent, old, "RIGHT", new, events)

    updates_after_step = total_updates(agent.q)

    # end_round appends to that very list, then calls end_of_round.
    events.append(e.SURVIVED_ROUND)
    train.end_of_round(agent, old, "RIGHT", events)

    assert total_updates(agent.q) == 1, (
        "the final transition must be learned from exactly once, not twice"
    )
    assert updates_after_step == 0, "n-step backup defers until the terminal flush"


def test_ql_survivor_still_receives_the_survival_bonus(ql_agent):
    """Deduplicating must not silently discard SURVIVED_ROUND's reward."""
    agent, train = ql_agent
    old, new = make_state(1), make_state(2, position=(2, 1))

    events = [e.MOVED_RIGHT]
    train.game_events_occurred(agent, old, "RIGHT", new, events)
    reward_before = agent._total_reward

    events.append(e.SURVIVED_ROUND)
    train.end_of_round(agent, old, "RIGHT", events)

    row = last_metrics_row(agent)
    bonus = agent.config.event_rewards[e.SURVIVED_ROUND]
    assert float(row["total_reward"]) == pytest.approx(reward_before + bonus, abs=1e-3)
    assert int(row["survived"]) == 1


def test_ql_death_path_learns_from_the_fatal_step(ql_agent):
    """A dead agent gets no game_events_occurred, so end_of_round must record it."""
    agent, train = ql_agent
    last = make_state(5, position=(3, 1))

    train.end_of_round(agent, last, "WAIT", [e.KILLED_SELF, e.GOT_KILLED])

    assert total_updates(agent.q) == 1

    row = last_metrics_row(agent)
    assert int(row["suicides"]) == 1
    # KILLED_SELF and GOT_KILLED both fire on a self-kill (environment.py:252
    # and 264), so the penalty is their sum.
    assert float(row["total_reward"]) < 0, "suicide must be penalised"


def test_ql_multi_step_episode_records_every_transition(ql_agent):
    agent, train = ql_agent

    states = [make_state(step, position=(1 + step % 2, 1)) for step in range(1, 7)]
    for index in range(len(states) - 1):
        train.game_events_occurred(agent, states[index], "RIGHT", states[index + 1], [e.MOVED_RIGHT])

    events = [e.MOVED_RIGHT, e.SURVIVED_ROUND]
    train.end_of_round(agent, states[-2], "RIGHT", events)

    # Five transitions were delivered; the last one arrived twice.
    assert total_updates(agent.q) == 5


def test_ql_end_of_round_without_a_state_is_safe(ql_agent):
    """end_round can fire with last_game_state None; it must not raise."""
    agent, train = ql_agent
    train.end_of_round(agent, None, None, [])
    assert total_updates(agent.q) == 0


def test_ql_handles_the_frameworks_substituted_actions(ql_agent):
    """A slow agent's action becomes 'WAIT', a crashed one's 'ERROR'."""
    agent, train = ql_agent
    old, new = make_state(1), make_state(2)

    train.game_events_occurred(agent, old, "ERROR", new, [e.INVALID_ACTION])
    train.end_of_round(agent, new, "ERROR", [])

    assert total_updates(agent.q) >= 1, "an unknown action must not crash the update"


def test_ql_round_index_follows_the_game(ql_agent):
    """callbacks.act and train both write round_index; they must agree."""
    agent, train = ql_agent
    state = make_state(1)
    state["round"] = 7

    train.end_of_round(agent, state, "WAIT", [e.SURVIVED_ROUND])

    assert agent.round_index == 7, "not an independent counter"


# --------------------------------------------------------------------------
# PPO
# --------------------------------------------------------------------------


@pytest.fixture
def ppo_agent(tmp_path, monkeypatch):
    from agent_code.attackontensor_ppo import callbacks, train

    # As above: isolate from the repository's real policy.pt, both on load and
    # on the atexit save.
    monkeypatch.setenv("AOT_PPO_MODEL_FILE", str(tmp_path / "policy.pt"))
    monkeypatch.setenv("AOT_PPO_METRICS_FILE", str(tmp_path / "metrics.csv"))
    monkeypatch.setenv("AOT_PPO_ROLLOUT_STEPS", "64")  # no update mid-test

    agent = make_self(train=True)
    callbacks.setup(agent)
    train.setup_training(agent)
    return agent, train


def test_ppo_survivor_final_transition_is_stored_once(ppo_agent):
    agent, train = ppo_agent
    old, new = make_state(1), make_state(2, position=(2, 1))

    events = [e.MOVED_RIGHT]
    train.game_events_occurred(agent, old, "RIGHT", new, events)
    assert len(agent.buffer) == 1

    events.append(e.SURVIVED_ROUND)
    train.end_of_round(agent, old, "RIGHT", events)

    assert len(agent.buffer) == 1, "the last transition must not be stored twice"


def test_ppo_survivor_transition_is_marked_terminal(ppo_agent):
    """GAE must not bootstrap across the episode boundary."""
    agent, train = ppo_agent
    old, new = make_state(1), make_state(2, position=(2, 1))

    events = [e.MOVED_RIGHT]
    train.game_events_occurred(agent, old, "RIGHT", new, events)
    assert not agent.buffer.dones[0], "not terminal while the round continues"

    events.append(e.SURVIVED_ROUND)
    train.end_of_round(agent, old, "RIGHT", events)

    assert agent.buffer.dones[0], "end_of_round must close the episode"


def test_ppo_survival_bonus_is_added_to_the_last_reward(ppo_agent):
    agent, train = ppo_agent
    old, new = make_state(1), make_state(2, position=(2, 1))

    events = [e.MOVED_RIGHT]
    train.game_events_occurred(agent, old, "RIGHT", new, events)
    reward_before = float(agent.buffer.rewards[0])

    events.append(e.SURVIVED_ROUND)
    train.end_of_round(agent, old, "RIGHT", events)

    bonus = agent.config.event_rewards[e.SURVIVED_ROUND]
    assert float(agent.buffer.rewards[0]) == pytest.approx(reward_before + bonus, abs=1e-4)


def test_ppo_death_path_stores_the_fatal_transition(ppo_agent):
    agent, train = ppo_agent
    last = make_state(5, position=(3, 1))

    train.end_of_round(agent, last, "WAIT", [e.KILLED_SELF, e.GOT_KILLED])

    assert len(agent.buffer) == 1
    assert agent.buffer.dones[0]
    assert agent.buffer.rewards[0] < 0


def test_ppo_act_caches_rollout_data_for_training(ppo_agent):
    """act() stashes the forward pass so train.py need not repeat it."""
    from agent_code.attackontensor_ppo import callbacks

    agent, _ = ppo_agent
    state = make_state(1)

    action = callbacks.act(agent, state)

    assert action in ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
    cached = agent._step_cache[(state["round"], state["step"])]
    assert cached["observation"].shape[0] == 13
    assert "log_prob" in cached and "value" in cached and "mask" in cached


def test_ql_round_stats_reset_every_round(ql_agent):
    """Per-round counters must not accumulate across rounds.

    The reset used to sit inside ``_write_snapshot``, which runs only every
    ``checkpoint_every`` rounds and returns early when snapshotting is off. The
    counters therefore summed over 250-round windows: every training curve came
    out a sawtooth, and ``survived`` latched to 1 for the rest of each window.
    Nothing errored, and the saved table was unaffected, so only the reported
    numbers were wrong.
    """
    agent, train = ql_agent
    # Snapshotting off and a checkpoint cadence this round is not a multiple of,
    # which is exactly the case that used to skip the reset entirely.
    agent.config.snapshot_dir = ""
    agent.config.checkpoint_every = 250

    arena = build_arena()
    for round_number in (1, 2, 3):
        state = make_state(1, coins=((1, 2),))
        train.game_events_occurred(
            agent, state, "DOWN", make_state(2, position=(1, 2)), [e.MOVED_DOWN, e.COIN_COLLECTED]
        )
        train.end_of_round(agent, make_state(2, position=(1, 2)), "DOWN", [e.SURVIVED_ROUND])
        agent.round_index = round_number

        row = last_metrics_row(agent)
        assert int(row["coins"]) == 1, (
            f"round {round_number} reported {row['coins']} coins; counters are accumulating"
        )
        # Two, because game_events_occurred and end_of_round each count a step.
        # Constant across rounds is the point: accumulating would give 2, 4, 6.
        assert int(row["steps"]) == 2, f"round {round_number} reported {row['steps']} steps"

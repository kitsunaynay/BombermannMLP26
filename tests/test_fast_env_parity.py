"""Fidelity of the fast training environment."""

import numpy as np
import pytest

import settings as s
from agent_code.scripted_test_agent.callbacks import script_action
from blib.fast_env import BomberEnv, FastWorld, world_args
from environment import BombeRLeWorld

AGENT_NAMES = ["scripted_test_agent", "scripted_test_agent_1"]


def build_real_world(seed, scenario, n_agents):
    """A stock BombeRLeWorld running the scripted fixture agent."""
    args = world_args(seed=seed, scenario=scenario, log_dir="logs")
    return BombeRLeWorld(args, [("scripted_test_agent", False)] * n_agents)


def build_fast_world(seed, scenario, names):
    def provide(states):
        return {
            name: script_action(state["round"], state["step"], name)
            for name, state in states.items()
        }

    return FastWorld(names, provide, scenario=scenario, seed=seed)


def snapshot(world):
    """Everything that defines the visible game state at one instant."""
    return {
        "step": world.step,
        "running": world.running,
        "arena": np.array(world.arena),
        "agents": sorted(
            (a.name, a.x, a.y, a.score, a.dead, a.bombs_left) for a in world.agents
        ),
        "bombs": sorted((b.x, b.y, b.timer) for b in world.bombs),
        "coins": sorted((c.x, c.y, c.collectable) for c in world.coins),
        "explosions": sorted(
            (tuple(sorted(x.blast_coords)), x.timer, x.stage) for x in world.explosions
        ),
    }


def assert_same(left, right, step):
    assert left["step"] == right["step"], f"step counter diverged at {step}"
    assert left["running"] == right["running"], f"running flag diverged at {step}"
    assert np.array_equal(left["arena"], right["arena"]), f"arena diverged at step {step}"
    assert left["agents"] == right["agents"], f"agent state diverged at step {step}"
    assert left["bombs"] == right["bombs"], f"bombs diverged at step {step}"
    assert left["coins"] == right["coins"], f"coins diverged at step {step}"
    assert left["explosions"] == right["explosions"], f"explosions diverged at step {step}"


@pytest.mark.parametrize("seed", [1, 7, 42])
@pytest.mark.parametrize("scenario", ["classic", "loot-crate", "coin-heaven"])
def test_fast_world_matches_the_real_world_step_for_step(seed, scenario):
    """The headline fidelity claim, over full rounds on every scenario."""
    n_agents = 2
    real = build_real_world(seed, scenario, n_agents)
    names = [a.name for a in real.agents]
    fast = build_fast_world(seed, scenario, names)

    real.new_round()
    fast.new_round()
    assert_same(snapshot(real), snapshot(fast), step=0)

    for step in range(1, s.MAX_STEPS + 1):
        if not real.running or not fast.running:
            break
        real.do_step()
        fast.do_step()
        assert_same(snapshot(real), snapshot(fast), step=step)

    assert real.running == fast.running
    assert real.step == fast.step, "rounds must end on the same step"


@pytest.mark.parametrize("seed", [3, 11])
def test_events_match_step_for_step(seed):
    """Rewards are derived from events, so the event streams must agree too."""
    real = build_real_world(seed, "classic", 2)
    names = [a.name for a in real.agents]
    fast = build_fast_world(seed, "classic", names)

    real.new_round()
    fast.new_round()

    for _ in range(60):
        if not real.running or not fast.running:
            break
        real.do_step()
        fast.do_step()

        real_events = {a.name: sorted(a.events) for a in real.agents}
        fast_events = {a.name: sorted(a.events) for a in fast.agents}
        assert real_events == fast_events


def test_fast_world_reproduces_itself_from_a_seed():
    names = ["a", "b"]
    first = snapshot_after(build_fast_world(5, "classic", names), 40)
    second = snapshot_after(build_fast_world(5, "classic", names), 40)

    assert np.array_equal(first["arena"], second["arena"])
    assert first["agents"] == second["agents"]


def snapshot_after(world, steps):
    world.new_round()
    for _ in range(steps):
        if not world.running:
            break
        world.do_step()
    return snapshot(world)


def test_different_seeds_give_different_arenas():
    names = ["a"]
    first = snapshot_after(build_fast_world(1, "classic", names), 1)
    second = snapshot_after(build_fast_world(2, "classic", names), 1)

    assert not np.array_equal(first["arena"], second["arena"])


# --------------------------------------------------------------------------
# BomberEnv wrapper
# --------------------------------------------------------------------------


def test_bomber_env_runs_an_episode_against_a_baseline():
    env = BomberEnv(opponents=["rule_based_agent"], scenario="classic", seed=3)
    state = env.reset()

    assert state is not None
    assert state["self"][0] == "learner"

    steps = 0
    done = False
    while not done and steps < s.MAX_STEPS:
        state, reward, done, info = env.step("WAIT")
        steps += 1
        assert isinstance(reward, float)
        assert "events" in info

    assert done, "episode must terminate within the step limit"
    assert env.round_statistics()["steps"] == steps


def test_bomber_env_reports_none_state_once_dead():
    """A dead learner gets None, matching environment.py:397."""
    env = BomberEnv(opponents=[], scenario="classic", seed=11)
    env.reset()

    # Bomb our own tile, then stand still until the blast lands.
    state, _, done, _ = env.step("BOMB")
    for _ in range(s.BOMB_TIMER + s.EXPLOSION_TIMER + 1):
        if done:
            break
        state, _, done, info = env.step("WAIT")

    assert done
    assert state is None
    assert not info["alive"]


def test_bomber_env_reward_function_receives_events():
    seen = []

    def reward_fn(old_state, action, new_state, events):
        seen.append(list(events))
        return 1.0

    env = BomberEnv(opponents=[], scenario="coin-heaven", seed=5, reward_fn=reward_fn)
    env.reset()
    _, reward, _, _ = env.step("RIGHT")

    assert reward == 1.0
    assert len(seen) == 1


def test_bomber_env_resets_between_episodes():
    env = BomberEnv(opponents=["peaceful_agent"], scenario="classic", seed=9)

    env.reset()
    for _ in range(5):
        env.step("WAIT")
    first_step_count = env.world.step

    env.reset()

    assert first_step_count > 0
    assert env.world.step == 0
    assert not env.learner.dead


def test_reward_receives_the_state_the_agent_acted_on():
    """`old_state` must be the observation the action was chosen against.

    do_step increments the step counter and only then polls, so reading
    last_game_state *before* do_step yields the previous step's observation.
    Pairing an action with a stale state is invisible in a training curve but
    corrupts every potential-based shaping term, which is a difference of two
    state potentials.
    """
    seen = []

    def reward_fn(old_state, action, new_state, events):
        seen.append((old_state, action, new_state))
        return 0.0

    env = BomberEnv(opponents=[], scenario="coin-heaven", seed=17, reward_fn=reward_fn)
    observed = [env.reset()]

    for _ in range(6):
        state, _, done, _ = env.step("WAIT")
        if done:
            break
        observed.append(state)

    for index, (old_state, _, new_state) in enumerate(seen[: len(observed) - 1]):
        assert old_state is not None
        # The state handed to the reward function is the one the agent saw when
        # it chose this action, and its successor is the next observation.
        assert old_state["step"] == index + 1, (
            f"transition {index}: old_state is from step {old_state['step']}"
        )
        if new_state is not None:
            assert new_state["step"] == old_state["step"]


def test_old_state_advances_by_one_step_each_call():
    states = []

    def reward_fn(old_state, action, new_state, events):
        states.append(old_state["step"] if old_state else None)
        return 0.0

    env = BomberEnv(opponents=[], scenario="coin-heaven", seed=23, reward_fn=reward_fn)
    env.reset()
    for _ in range(5):
        _, _, done, _ = env.step("WAIT")
        if done:
            break

    assert states == list(range(1, len(states) + 1)), f"steps went {states}"

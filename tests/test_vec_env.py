"""Vectorised environment tests.

The subtle contract here is auto-reset: when an episode ends the worker returns
the *next* episode's first observation, and stashes the true terminal
observation in ``info``. A learner that bootstraps from the reset observation
instead corrupts its value targets at every episode boundary, and nothing about
the training curve makes that obvious.
"""

import numpy as np
import pytest

from blib.seeding import derive_seed, episode_seed, evaluation_seeds, worker_seed
from blib.vec_env import DummyVecEnv, EnvSpec, SubprocVecEnv, make_specs


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------


def test_seed_derivation_is_stable_and_distinct():
    assert derive_seed(7, "worker", 3) == derive_seed(7, "worker", 3)
    assert derive_seed(7, "worker", 3) != derive_seed(7, "worker", 4)
    assert derive_seed(7, "worker", 3) != derive_seed(8, "worker", 3)
    assert 0 <= derive_seed(7, "a") < 2**32


def test_worker_and_episode_seeds_do_not_collide():
    seeds = {worker_seed(1, i) for i in range(64)}
    assert len(seeds) == 64

    episodes = {episode_seed(1, 0, i) for i in range(64)}
    assert len(episodes) == 64


def test_evaluation_seeds_are_fixed_across_calls():
    """Every checkpoint must be scored on identical arenas."""
    assert evaluation_seeds(3, 10) == evaluation_seeds(3, 10)
    assert len(set(evaluation_seeds(3, 10))) == 10


# --------------------------------------------------------------------------
# Vectorised stepping
# --------------------------------------------------------------------------


def test_dummy_vec_env_steps_every_environment():
    specs = make_specs(3, base_seed=1, scenario="coin-heaven")
    with DummyVecEnv(specs) as env:
        observations = env.reset()
        assert len(observations) == 3

        observations, rewards, dones, infos = env.step(["WAIT"] * 3)

        assert len(observations) == 3
        assert rewards.shape == (3,)
        assert dones.shape == (3,)
        assert len(infos) == 3


def test_specs_get_distinct_seeds():
    """Identical seeds would make N workers replay one arena N times."""
    specs = make_specs(4, base_seed=5)
    assert len({spec.seed for spec in specs}) == 4


def test_auto_reset_preserves_the_terminal_observation():
    specs = [EnvSpec(opponents=(), scenario="classic", seed=11)]
    with DummyVecEnv(specs) as env:
        env.reset()

        # Bomb our own tile and wait: a guaranteed, quick death.
        actions = ["BOMB"] + ["WAIT"] * 12
        for action in actions:
            observations, rewards, dones, infos = env.step([action])
            if dones[0]:
                break

        assert dones[0], "the episode should have ended"
        assert "terminal_observation" in infos[0]
        assert "round_statistics" in infos[0]
        # The returned observation belongs to the *new* episode, not the old one.
        assert observations[0] is not None
        assert observations[0]["step"] == 0


def test_transform_is_applied_inside_the_worker():
    def transform_factory():
        return lambda state: {"step": state["step"], "encoded": True}

    specs = [EnvSpec(scenario="coin-heaven", seed=2, transform_factory=transform_factory)]
    with DummyVecEnv(specs) as env:
        observation = env.reset()[0]
        assert observation["encoded"] is True


def test_step_rejects_a_wrong_action_count():
    with DummyVecEnv(make_specs(2, base_seed=1)) as env:
        env.reset()
        with pytest.raises(ValueError, match="expected 2 actions"):
            env.step(["WAIT"])


def test_empty_spec_list_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        SubprocVecEnv([])


@pytest.mark.slow
def test_subproc_vec_env_matches_the_dummy_implementation():
    """The process-backed and in-process versions must agree.

    Spawning processes is slow, so this runs once with a tiny configuration --
    enough to prove the pipe protocol and pickling work end to end.
    """
    specs = make_specs(2, base_seed=4, scenario="coin-heaven")

    with DummyVecEnv(specs) as reference:
        reference.reset()
        _, dummy_rewards, _, _ = reference.step(["RIGHT", "RIGHT"])

    with SubprocVecEnv(specs) as parallel:
        parallel.reset()
        observations, rewards, dones, _ = parallel.step(["RIGHT", "RIGHT"])

    assert len(observations) == 2
    assert np.array_equal(rewards, dummy_rewards)
    assert dones.shape == (2,)

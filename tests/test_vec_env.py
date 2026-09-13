"""Vectorised environment tests."""

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


# --------------------------------------------------------------------------
# Opponent specs (blib.opponents.OpponentSpec)
# --------------------------------------------------------------------------


def test_opponent_spec_parses_modifiers(tmp_path):
    from blib.opponents import OpponentSpec, display_name

    plain = OpponentSpec.parse("rule_based_agent")
    assert (plain.code_name, plain.model_file, plain.no_bomb) == ("rule_based_agent", None, False)

    nobomb = OpponentSpec.parse("rule_based_agent:nobomb")
    assert nobomb.no_bomb and nobomb.code_name == "rule_based_agent"
    assert display_name("rule_based_agent:nobomb") == "rule_based_agent-nobomb"

    frozen = tmp_path / "snap.pt"
    spec = OpponentSpec.parse(f"attackontensor_ppo@{frozen}:nobomb")
    assert spec.code_name == "attackontensor_ppo"
    assert spec.model_file == str(frozen.resolve())
    assert spec.no_bomb
    assert spec.display_name == "attackontensor_ppo@snap-nobomb"


def test_nobomb_opponent_never_bombs():
    """The provided agent's policy is untouched; only its BOMB is swapped for WAIT."""
    from blib.fast_env import FastWorld
    from blib.opponents import ScriptedOpponent

    plain = ScriptedOpponent("rule_based_agent")
    muted = ScriptedOpponent("rule_based_agent:nobomb")
    assert muted.module is plain.module

    # Let the unmodified agent drive the probe so it reaches crates and bombs;
    # the muted twin sees the same states and must never answer BOMB.
    plain_actions, muted_actions = [], []

    def provide(states):
        state = states["probe"]
        plain_actions.append(plain.act(state))
        muted_actions.append(muted.act(state))
        return {"probe": plain_actions[-1]}

    world = FastWorld(["probe"], provide, scenario="classic", seed=3)
    world.new_round()
    for _ in range(120):
        if not world.running:
            break
        world.do_step()

    assert "BOMB" in plain_actions, "rule_based_agent never bombed in 120 steps"
    assert "BOMB" not in muted_actions
    assert len(muted_actions) == len(plain_actions)


def test_frozen_checkpoint_spec_is_checked_and_restores_the_environment(tmp_path, monkeypatch):
    import os
    import pytest
    from blib.opponents import ScriptedOpponent

    with pytest.raises(FileNotFoundError):
        ScriptedOpponent(f"attackontensor_ppo@{tmp_path / 'missing.pt'}")

    with pytest.raises(ValueError):
        ScriptedOpponent(f"rule_based_agent@{tmp_path / 'x.pt'}")

    # A real (untrained) checkpoint written by the learner loads through the
    # spec, and the learner's own model path is untouched afterwards.
    import torch
    from agent_code.attackontensor_ppo import tensorizer as T
    from agent_code.attackontensor_ppo.config import PPOConfig
    from agent_code.attackontensor_ppo.network import build_network
    from agent_code.attackontensor_ppo.ppo import PPOLearner

    config = PPOConfig()
    PPOLearner(build_network(config, T.n_channels(config)), config).save(tmp_path / "frozen.pt")
    monkeypatch.setenv("AOT_PPO_MODEL_FILE", "learner-own.pt")
    opponent = ScriptedOpponent(f"attackontensor_ppo@{tmp_path / 'frozen.pt'}")
    assert os.environ["AOT_PPO_MODEL_FILE"] == "learner-own.pt"
    assert opponent.state.config.model_path == tmp_path / "frozen.pt" or \
        str(opponent.state.config.model_path).endswith("frozen.pt")

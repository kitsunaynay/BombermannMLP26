"""Tests for the PPO agent."""

import numpy as np
import pytest
import torch

import settings as s
from agent_code.attackontensor_ppo import tensorizer as T
from agent_code.attackontensor_ppo.config import PPOConfig
from agent_code.attackontensor_ppo.kit.actions import N_ACTIONS
from agent_code.attackontensor_ppo.network import MASK_FILL, ActorCritic, build_network
from agent_code.attackontensor_ppo.ppo import (
    PPOLearner,
    RolloutBuffer,
    compute_gae,
    explained_variance,
)
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


# --------------------------------------------------------------------------
# Tensorizer
# --------------------------------------------------------------------------


def test_tensor_has_the_documented_shape_and_dtype():
    config = PPOConfig(observation="global")
    tensor = T.state_to_tensor(make_game_state(), config)

    assert tensor.shape == (T.BASE_CHANNELS, s.COLS, s.ROWS)
    assert tensor.dtype == np.float32
    assert len(T.CHANNEL_NAMES) == T.N_CHANNELS == T.BASE_CHANNELS + T.SURVIVAL_CHANNELS


def test_survival_channels_are_opt_in():
    state = make_game_state(bombs=[((1, 1), 3)])
    base = T.state_to_tensor(state, PPOConfig(survival_channels=False))
    full = T.state_to_tensor(state, PPOConfig(survival_channels=True))

    assert base.shape[0] == T.BASE_CHANNELS and full.shape[0] == T.N_CHANNELS
    # The base planes are identical either way; only the extra ones differ.
    assert np.array_equal(base, full[: T.BASE_CHANNELS])
    assert full[T.CH_SURVIVAL_DURATION:].any()


def test_tensor_is_none_for_a_missing_state():
    assert T.state_to_tensor(None, PPOConfig()) is None


def test_terrain_channels_partition_the_board():
    """Every tile is exactly one of wall, crate, or free."""
    tensor = T.state_to_tensor(make_game_state(), PPOConfig(observation="global"))
    total = tensor[T.CH_WALL] + tensor[T.CH_CRATE] + tensor[T.CH_FREE]
    assert np.all(total == 1.0)


def test_entity_channels_are_placed_correctly():
    config = PPOConfig(observation="global")
    state = make_game_state(position=(1, 1), coins=[(3, 1)], others=[(1, 3)])
    tensor = T.state_to_tensor(state, config)

    assert tensor[T.CH_SELF, 1, 1] == 1.0
    assert tensor[T.CH_SELF].sum() == 1.0
    assert tensor[T.CH_COIN, 3, 1] == 1.0
    assert tensor[T.CH_OTHERS, 1, 3] == 1.0
    assert tensor[T.CH_BOMB_READY].min() == 1.0, "broadcast plane, agent has its bomb"


def test_danger_channel_is_largest_for_imminent_blasts():
    """Channel 8 inverts the timer, so 1.0 means 'lethal at the end of this step'."""
    field = np.full((9, 9), -1, dtype=int)
    field[1:8, 1] = 0
    config = PPOConfig(observation="global")

    urgent = T.state_to_tensor(make_game_state(field, (5, 1), bombs=[((1, 1), 0)]), config)
    distant = T.state_to_tensor(make_game_state(field, (5, 1), bombs=[((1, 1), 3)]), config)

    assert urgent[T.CH_DANGER, 2, 1] == pytest.approx(1.0)
    assert distant[T.CH_DANGER, 2, 1] < urgent[T.CH_DANGER, 2, 1]
    assert np.all(urgent[T.CH_DANGER] >= 0.0) and np.all(urgent[T.CH_DANGER] <= 1.0)


def test_egocentric_crop_is_centred_and_wall_padded():
    config = PPOConfig(observation="ego", ego_radius=4)
    tensor = T.state_to_tensor(make_game_state(position=(1, 1)), config)

    size = 2 * config.ego_radius + 1
    assert tensor.shape == (T.n_channels(config), size, size)
    # The agent sits at the centre of its own view, by construction.
    assert tensor[T.CH_SELF, config.ego_radius, config.ego_radius] == 1.0
    assert tensor[T.CH_SELF].sum() == 1.0
    # Standing at (1, 1) means the window overhangs the board; that is wall.
    assert tensor[T.CH_WALL, 0, 0] == 1.0


def test_observation_shape_matches_the_tensorizer():
    for config in (
        PPOConfig(observation="global"),
        PPOConfig(observation="ego", ego_radius=3),
        PPOConfig(observation="global", survival_channels=True),
    ):
        assert T.state_to_tensor(make_game_state(), config).shape == T.observation_shape(config)


# --------------------------------------------------------------------------
# GAE -- checked against an independent reference
# --------------------------------------------------------------------------


def reference_gae(rewards, values, dones, last_value, gamma, lam):
    """Direct transcription of the GAE sum, written independently of the code."""
    steps = len(rewards)
    deltas = np.zeros(steps)
    for t in range(steps):
        next_value = last_value if t == steps - 1 else values[t + 1]
        if dones[t]:
            next_value = 0.0
        deltas[t] = rewards[t] + gamma * next_value - values[t]

    advantages = np.zeros(steps)
    for t in range(steps):
        total = 0.0
        weight = 1.0
        for k in range(t, steps):
            total += weight * deltas[k]
            if dones[k]:
                break
            weight *= gamma * lam
        advantages[t] = total
    return advantages


@pytest.mark.parametrize("seed", range(5))
def test_gae_matches_the_reference_sum(seed):
    rng = np.random.default_rng(seed)
    steps = 24
    rewards = rng.normal(size=steps)
    values = rng.normal(size=steps)
    dones = rng.random(steps) < 0.2
    last_value = float(rng.normal())

    advantages, returns = compute_gae(rewards, values, dones, last_value, 0.99, 0.95)
    expected = reference_gae(rewards, values, dones, last_value, 0.99, 0.95)

    assert np.allclose(advantages, expected)
    assert np.allclose(returns, advantages + values)


def test_gae_does_not_bootstrap_across_an_episode_boundary():
    """A `done` must cut both the bootstrap and the recursion.

    If it did not, credit would leak from one episode into the previous one,
    which is the classic silent GAE bug.
    """
    rewards = np.array([1.0, 0.0])
    values = np.array([0.0, 0.0])

    cut = compute_gae(rewards, values, np.array([True, False]), 100.0, 0.99, 0.95)[0]
    joined = compute_gae(rewards, values, np.array([False, False]), 100.0, 0.99, 0.95)[0]

    assert cut[0] == pytest.approx(1.0), "terminal step sees only its own reward"
    assert joined[0] != pytest.approx(1.0), "without the cut, future value leaks in"


def test_gae_with_lambda_one_is_the_monte_carlo_return():
    rewards = np.array([1.0, 2.0, 3.0])
    values = np.zeros(3)
    dones = np.array([False, False, True])

    _, returns = compute_gae(rewards, values, dones, 0.0, 0.5, 1.0)

    assert returns[0] == pytest.approx(1 + 0.5 * 2 + 0.25 * 3)
    assert returns[2] == pytest.approx(3.0)


def test_explained_variance_bounds():
    targets = np.array([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(targets, targets) == pytest.approx(1.0)
    assert explained_variance(np.full(4, targets.mean()), targets) == pytest.approx(0.0)
    assert explained_variance(np.zeros(4), np.zeros(4)) == 0.0, "constant target is safe"


# --------------------------------------------------------------------------
# Rollout buffer
# --------------------------------------------------------------------------


def test_buffer_stores_and_reports_fullness():
    buffer = RolloutBuffer(capacity=3, observation_shape=(2, 4, 4))
    observation = np.zeros((2, 4, 4), dtype=np.float32)

    for index in range(3):
        buffer.add(observation, index, -0.5, 1.0, 0.25, False)

    assert len(buffer) == 3 and buffer.is_full
    with pytest.raises(RuntimeError, match="full"):
        buffer.add(observation, 0, 0.0, 0.0, 0.0, False)

    buffer.reset()
    assert len(buffer) == 0 and not buffer.is_full


def test_buffer_defaults_masks_to_all_true():
    buffer = RolloutBuffer(capacity=2, observation_shape=(1, 2, 2))
    buffer.add(np.zeros((1, 2, 2), dtype=np.float32), 0, 0.0, 0.0, 0.0, False, mask=None)
    assert buffer.masks[0].all()


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


def test_network_output_shapes():
    network = ActorCritic(in_channels=T.N_CHANNELS, spatial_size=17)
    logits, value = network(torch.zeros(5, T.N_CHANNELS, 17, 17))

    assert logits.shape == (5, N_ACTIONS)
    assert value.shape == (5,)


def test_network_adapts_to_the_egocentric_size():
    config = PPOConfig(observation="ego", ego_radius=3)
    network = build_network(config, T.n_channels(config))
    logits, _ = network(torch.zeros(2, T.n_channels(config), 7, 7))
    assert logits.shape == (2, N_ACTIONS)


def test_masked_actions_are_never_sampled():
    network = ActorCritic(in_channels=T.N_CHANNELS, spatial_size=9)
    observation = torch.randn(1, T.N_CHANNELS, 9, 9)
    mask = torch.zeros(1, N_ACTIONS, dtype=torch.bool)
    mask[0, 2] = True

    for _ in range(50):
        action, _, _ = network.act(observation, mask)
        assert int(action.item()) == 2


def test_an_all_false_mask_does_not_produce_nan():
    """When death is certain the mask can be empty; softmax must still be finite."""
    network = ActorCritic(in_channels=T.N_CHANNELS, spatial_size=9)
    observation = torch.randn(1, T.N_CHANNELS, 9, 9)
    mask = torch.zeros(1, N_ACTIONS, dtype=torch.bool)

    logits, value = network(observation, mask)
    action, log_prob, _ = network.act(observation, mask)

    assert torch.isfinite(logits).all()
    assert torch.isfinite(log_prob).all() and torch.isfinite(value).all()
    assert (logits != MASK_FILL).any()


def test_policy_starts_near_uniform():
    """gain=0.01 on the policy head keeps initial entropy high.

    A confident-but-arbitrary initial policy collapses entropy before any
    learning happens, which is the characteristic way PPO fails here.
    """
    network = ActorCritic(in_channels=T.N_CHANNELS, spatial_size=17)
    logits, _ = network(torch.randn(32, T.N_CHANNELS, 17, 17))
    entropy = torch.distributions.Categorical(logits=logits).entropy().mean()

    assert entropy > 0.95 * np.log(N_ACTIONS)


# --------------------------------------------------------------------------
# Learner
# --------------------------------------------------------------------------


def test_update_runs_and_reports_finite_metrics():
    config = PPOConfig(rollout_steps=64, minibatch_size=16, update_epochs=2, observation="ego", ego_radius=2)
    network = build_network(config, T.n_channels(config))
    learner = PPOLearner(network, config)

    shape = T.observation_shape(config)
    buffer = RolloutBuffer(capacity=64, observation_shape=shape)
    rng = np.random.default_rng(0)
    for _ in range(64):
        buffer.add(
            rng.standard_normal(shape).astype(np.float32),
            int(rng.integers(N_ACTIONS)),
            float(np.log(1 / N_ACTIONS)),
            float(rng.normal()),
            float(rng.normal()),
            False,
        )

    advantages, returns = buffer.compute_advantages(0.0, config.gamma, config.gae_lambda)
    metrics = learner.update(buffer, advantages, returns)

    assert metrics.n_updates > 0
    for name, value in metrics.as_dict().items():
        assert np.isfinite(value), f"{name} is not finite"
    assert metrics.approx_kl >= 0.0, "the k3 estimator is non-negative by construction"


def test_update_on_an_empty_buffer_is_a_no_op():
    config = PPOConfig(observation="ego", ego_radius=2)
    learner = PPOLearner(build_network(config, T.n_channels(config)), config)
    buffer = RolloutBuffer(capacity=4, observation_shape=T.observation_shape(config))

    metrics = learner.update(buffer, np.array([]), np.array([]))

    assert metrics.n_updates == 0


def test_checkpoint_round_trips(tmp_path):
    config = PPOConfig(observation="ego", ego_radius=2, survival_channels=True)
    learner = PPOLearner(build_network(config, T.n_channels(config)), config)
    path = tmp_path / "policy.pt"

    learner.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["network"]["in_channels"] == T.N_CHANNELS
    assert payload["config"]["survival_channels"] is True
    assert payload["network"]["spatial_size"] == config.spatial_size
    assert "state_dict" in payload and "config" in payload


def test_schedules_anneal_between_the_configured_endpoints():
    config = PPOConfig(learning_rate=1e-3, learning_rate_final=1e-5, anneal_schedules=True)
    learner = PPOLearner(build_network(config, T.n_channels(config)), config)

    learner.progress = 0.0
    assert learner.learning_rate == pytest.approx(1e-3)
    learner.progress = 1.0
    assert learner.learning_rate == pytest.approx(1e-5)
    learner.progress = 0.5
    assert 1e-5 < learner.learning_rate < 1e-3


def test_evaluation_takes_the_argmax_by_default():
    """The shipped default is greedy, and the training path still samples.

    History: with the high-entropy 250k-step Task-1 checkpoint, sampling scored
    31.6 coins [29.6, 33.5] against 13.4 [10.2, 16.7] for argmax, so the default
    was stochastic for Phases 6-19. Re-measured on the sharp stage-4 policy
    (2026-09-16, 700 paired arenas vs rule_based x3): argmax 9.51 vs sampling
    8.79, +0.72 [+0.23, +1.20], suicides 11.4% vs 15.7%, and ties in 1v1 and
    the 4-player pool. The guard now pins the new default; the second assert
    pins that rollouts (self.train) are unaffected, since PPO needs samples.
    """
    assert PPOConfig().deterministic_eval is True
    import agent_code.attackontensor_ppo.callbacks as C
    import inspect
    assert "(not self.train) and self.config.deterministic_eval" in inspect.getsource(C)


def test_deterministic_flag_actually_switches_behaviour():
    network = ActorCritic(in_channels=T.N_CHANNELS, spatial_size=9)
    observation = torch.randn(1, T.N_CHANNELS, 9, 9)

    greedy = {int(network.act(observation, deterministic=True)[0].item()) for _ in range(20)}
    sampled = {int(network.act(observation, deterministic=False)[0].item()) for _ in range(60)}

    assert len(greedy) == 1, "argmax must be deterministic"
    assert len(sampled) > 1, "sampling must explore the action distribution"

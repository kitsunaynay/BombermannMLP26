"""Widening a checkpoint's input must not change the function it computes."""

import logging
from types import SimpleNamespace

import numpy as np
import torch

from agent_code.attackontensor_ppo import callbacks
from agent_code.attackontensor_ppo import tensorizer as T
from agent_code.attackontensor_ppo.config import PPOConfig
from agent_code.attackontensor_ppo.network import build_network
from agent_code.attackontensor_ppo.ppo import PPOLearner
from tools.widen_checkpoint import FIRST_CONV_WEIGHT, widen


def _trained_like_checkpoint(tmp_path):
    config = PPOConfig(survival_channels=False, safety_mode="hard")
    network = build_network(config, T.n_channels(config))
    torch.manual_seed(1)
    with torch.no_grad():  # non-trivial weights, so equality is a real check
        for parameter in network.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    path = tmp_path / "base.pt"
    PPOLearner(network, config).save(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def test_widened_checkpoint_computes_the_same_logits(tmp_path):
    payload = _trained_like_checkpoint(tmp_path)
    widened = widen(payload, T.N_CHANNELS)

    assert widened["state_dict"][FIRST_CONV_WEIGHT].shape[1] == T.N_CHANNELS
    assert widened["network"]["in_channels"] == T.N_CHANNELS
    assert widened["config"]["survival_channels"] is True
    assert widened["widened_from_channels"] == T.BASE_CHANNELS

    narrow = build_network(PPOConfig(survival_channels=False), T.BASE_CHANNELS)
    narrow.load_state_dict(payload["state_dict"])
    wide = build_network(PPOConfig(survival_channels=True), T.N_CHANNELS)
    wide.load_state_dict(widened["state_dict"])

    observation = torch.rand(4, T.N_CHANNELS, 17, 17)
    with torch.no_grad():
        logits_narrow, value_narrow = narrow(observation[:, : T.BASE_CHANNELS])
        logits_wide, value_wide = wide(observation)
    assert torch.allclose(logits_narrow, logits_wide, atol=1e-6)
    assert torch.allclose(value_narrow, value_wide, atol=1e-6)


def test_widened_checkpoint_loads_through_setup(tmp_path, monkeypatch):
    payload = _trained_like_checkpoint(tmp_path)
    path = tmp_path / "wide.pt"
    torch.save(widen(payload, T.N_CHANNELS), path)
    monkeypatch.setenv("AOT_PPO_MODEL_FILE", str(path))

    agent = SimpleNamespace(logger=logging.getLogger("test_widen"), train=False)
    callbacks.setup(agent)
    assert agent.config.survival_channels is True
    assert agent.config.safety_mode == "hard"
    assert agent.network.in_channels == T.N_CHANNELS
    stored = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    assert torch.equal(agent.network.state_dict()[FIRST_CONV_WEIGHT], stored[FIRST_CONV_WEIGHT])


def test_widen_refuses_to_shrink(tmp_path):
    import pytest
    payload = _trained_like_checkpoint(tmp_path)
    wide = widen(payload, T.N_CHANNELS)
    with pytest.raises(ValueError):
        widen(wide, T.BASE_CHANNELS)
    assert np.array_equal(
        widen(payload, T.BASE_CHANNELS)["state_dict"][FIRST_CONV_WEIGHT].numpy(),
        payload["state_dict"][FIRST_CONV_WEIGHT].numpy(),
    )

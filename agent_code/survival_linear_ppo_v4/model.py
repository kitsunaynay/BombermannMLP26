"""Linear PPO actor-critic with resumable optimizer state."""
from __future__ import annotations

import os
from pathlib import Path
import pickle

import numpy as np

from .core import ACTIONS, FEATURE_NAMES, STATE_FEATURE_NAMES, safe_action_data, state_features

DEFAULT_CONFIG = {
    "gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.20,
    "actor_lr": 4e-3, "critic_lr": 3e-3, "entropy_coef": 0.010,
    "ppo_epochs": 4, "minibatch_size": 256, "min_update_steps": 256,
    "target_kl": 0.012, "max_grad_norm": 0.75, "seed": 7301,
}


def softmax_masked(logits, mask):
    logits = np.where(mask, logits, -1e9)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(np.clip(logits, -60, 60)) * mask
    return exponentials / np.maximum(exponentials.sum(axis=-1, keepdims=True), 1e-12)


class Adam:
    def __init__(self, shape, lr):
        self.lr = float(lr)
        self.initial_lr = float(lr)
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.t = 0

    def step(self, parameter, gradient, max_norm):
        norm = float(np.linalg.norm(gradient))
        if norm > max_norm:
            gradient = gradient * max_norm / (norm + 1e-12)
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * gradient
        self.v = 0.999 * self.v + 0.001 * gradient * gradient
        m_hat = self.m / (1 - 0.9 ** self.t)
        v_hat = self.v / (1 - 0.999 ** self.t)
        parameter += self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def state_dict(self):
        return {"lr": self.lr, "initial_lr": self.initial_lr, "m": self.m, "v": self.v, "t": self.t}

    def load_state_dict(self, state):
        if not state:
            return
        self.lr = float(state["lr"])
        self.initial_lr = float(state.get("initial_lr", self.initial_lr))
        self.m[:] = state["m"]
        self.v[:] = state["v"]
        self.t = int(state["t"])


class LinearPPO:
    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.theta = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        self.value_w = np.zeros(len(STATE_FEATURE_NAMES), dtype=np.float64)
        priors = {
            "survival_breadth": 1.2, "terminal_width": 0.5, "blast_margin": 0.5,
            "survival_duration": 1.0, "danger_improvement": 0.8,
            "coin_progress": 0.35, "crate_progress": 0.25, "opponent_progress": 0.10,
            "free_neighbors": 0.20, "corridor_exit": 0.35, "dead_end_risk": -0.8,
            "opponent_pressure": -0.10, "contested_now": -1.0, "contested_path": -0.5,
            "visit_penalty": -0.25, "reverse_action": -0.20, "plan_match": 0.75,
            "own_blast_exit": 1.25, "move": 0.10, "safe_wait": -0.05, "bomb": -0.30,
            "bomb_crates": 0.60, "bomb_opponents": 1.20, "bomb_has_target": 0.25,
            "bomb_escape_speed": 0.60, "bomb_escape_breadth": 0.75,
            "future_bomb_value": 0.25, "trap_opportunity": 0.50,
        }
        for name, value in priors.items():
            self.theta[FEATURE_NAMES.index(name)] = value
        self.n_updates = 0
        self.actor_opt = Adam(self.theta.shape, self.config["actor_lr"])
        self.critic_opt = Adam(self.value_w.shape, self.config["critic_lr"])

    def probs(self, phi, mask):
        return softmax_masked(np.einsum("...ad,d->...a", phi, self.theta), mask)

    def value(self, state_vector):
        return np.asarray(state_vector) @ self.value_w

    def choose(self, game_state, rng, stochastic=True, context=None):
        context = {**(context or {}), "shield_mode": self.config.get("shield_mode", "full")}
        phi, mask, metadata = safe_action_data(game_state, context)
        state_vector = state_features(game_state, context, (phi, mask, metadata))
        probabilities = self.probs(phi, mask)
        action_index = int(rng.choice(len(ACTIONS), p=probabilities)) if stochastic else int(np.argmax(probabilities))
        record = {
            "phi": phi, "mask": mask, "state": state_vector,
            "action": action_index, "old_logp": float(np.log(probabilities[action_index] + 1e-12)),
            "old_probs": probabilities.copy(), "value": float(self.value(state_vector)),
            "reward": None, "next_value": 0.0, "done": False,
        }
        decision = {
            "certificate": metadata["certificates"][action_index],
            "fallback": metadata["fallback"],
        }
        return ACTIONS[action_index], record, decision

    def _gae(self, records):
        rewards = np.asarray([item["reward"] for item in records], dtype=float)
        values = np.asarray([item["value"] for item in records], dtype=float)
        next_values = np.asarray([item["next_value"] for item in records], dtype=float)
        dones = np.asarray([item["done"] for item in records], dtype=float)
        deltas = rewards + self.config["gamma"] * next_values * (1 - dones) - values
        advantages = np.zeros_like(deltas)
        running = 0.0
        for index in range(len(records) - 1, -1, -1):
            running = deltas[index] + self.config["gamma"] * self.config["gae_lambda"] * (1 - dones[index]) * running
            advantages[index] = running
        returns = np.clip(advantages + values, -20.0, 20.0)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def _exact_kl(self, phi, mask, old_probs):
        new_probs = self.probs(phi, mask)
        return float(np.mean(np.sum(old_probs * (np.log(old_probs + 1e-12) - np.log(new_probs + 1e-12)), axis=1)))

    def update(self, records, rng):
        records = [item for item in records if item["reward"] is not None]
        if not records:
            return {}
        phi = np.stack([item["phi"] for item in records])
        mask = np.stack([item["mask"] for item in records])
        state = np.stack([item["state"] for item in records])
        actions = np.asarray([item["action"] for item in records], dtype=int)
        old_logp = np.asarray([item["old_logp"] for item in records])
        old_probs = np.stack([item["old_probs"] for item in records])
        advantages, returns = self._gae(records)
        n_items = len(records)
        last_kl = 0.0
        clip_fractions = []

        for _ in range(self.config["ppo_epochs"]):
            epoch_start = self.theta.copy()
            shuffled = rng.permutation(n_items)
            for start in range(0, n_items, self.config["minibatch_size"]):
                indices = shuffled[start:start + self.config["minibatch_size"]]
                probabilities = self.probs(phi[indices], mask[indices])
                chosen = probabilities[np.arange(len(indices)), actions[indices]]
                ratio = np.exp(np.log(chosen + 1e-12) - old_logp[indices])
                advantage = advantages[indices]
                clipped = ((advantage >= 0) & (ratio > 1 + self.config["clip_epsilon"])) | ((advantage < 0) & (ratio < 1 - self.config["clip_epsilon"]))
                clip_fractions.append(float(clipped.mean()))
                expected_phi = np.einsum("ba,bad->bd", probabilities, phi[indices])
                score_gradient = phi[indices, actions[indices]] - expected_phi
                actor_gradient = np.mean((~clipped)[:, None] * ratio[:, None] * advantage[:, None] * score_gradient, axis=0)
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
                entropy_logit_gradient = -probabilities * (np.log(probabilities + 1e-12) + entropy[:, None])
                actor_gradient += self.config["entropy_coef"] * np.mean(
                    np.einsum("ba,bad->bd", entropy_logit_gradient, phi[indices]), axis=0
                )
                self.actor_opt.step(self.theta, actor_gradient, self.config["max_grad_norm"])

                prediction = state[indices] @ self.value_w
                critic_error = np.clip(returns[indices] - prediction, -1.0, 1.0)  # Huber gradient
                critic_gradient = np.mean(critic_error[:, None] * state[indices], axis=0)
                self.critic_opt.step(self.value_w, critic_gradient, self.config["max_grad_norm"])

            last_kl = self._exact_kl(phi, mask, old_probs)
            if last_kl > 1.5 * self.config["target_kl"]:
                candidate = self.theta.copy()
                for fraction in (0.5, 0.25, 0.125, 0.0):
                    self.theta[:] = epoch_start + fraction * (candidate - epoch_start)
                    if self._exact_kl(phi, mask, old_probs) <= self.config["target_kl"]:
                        break
                last_kl = self._exact_kl(phi, mask, old_probs)
                break

        if last_kl > 1.25 * self.config["target_kl"]:
            self.actor_opt.lr = max(self.actor_opt.initial_lr * 0.10, self.actor_opt.lr * 0.70)
        elif last_kl < 0.50 * self.config["target_kl"]:
            self.actor_opt.lr = min(self.actor_opt.initial_lr, self.actor_opt.lr * 1.03)
        self.n_updates += 1
        final_probs = self.probs(phi, mask)
        final_entropy = float(np.mean(-np.sum(final_probs * np.log(final_probs + 1e-12), axis=1)))
        return {
            "update": self.n_updates, "steps": n_items,
            "reward_mean": float(np.mean([item["reward"] for item in records])),
            "return_mean": float(returns.mean()),
            "value_mse": float(np.mean((state @ self.value_w - returns) ** 2)),
            "entropy": final_entropy, "kl": last_kl,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "actor_lr": self.actor_opt.lr,
        }

    def save(self, path):
        path = Path(path)
        payload = {
            "theta": self.theta, "value_w": self.value_w, "config": self.config,
            "n_updates": self.n_updates, "feature_names": FEATURE_NAMES,
            "state_feature_names": STATE_FEATURE_NAMES,
            "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
            "format_version": 4,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as stream:
            payload = pickle.load(stream)
        if payload.get("feature_names") != FEATURE_NAMES or payload.get("state_feature_names") != STATE_FEATURE_NAMES:
            raise ValueError("Saved V4 feature schema does not match this code")
        model = cls(payload.get("config"))
        model.theta[:] = payload["theta"]
        model.value_w[:] = payload["value_w"]
        model.n_updates = int(payload.get("n_updates", 0))
        model.actor_opt.load_state_dict(payload.get("actor_opt"))
        model.critic_opt.load_state_dict(payload.get("critic_opt"))
        return model

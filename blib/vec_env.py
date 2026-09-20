"""Parallel environment."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .seeding import worker_seed

# Worker protocol
CMD_RESET = "reset"
CMD_STEP = "step"
CMD_STATS = "stats"
CMD_CLOSE = "close"


@dataclass
class EnvSpec:
    """Everything needed to build an environment in a child process.

    A plain dataclass rather than a closure so it pickles cleanly under the
    ``spawn`` start method, which is the default on macOS and the safe choice
    everywhere.
    """

    opponents: Sequence[str] = ()
    scenario: str = "classic"
    seed: Optional[int] = None
    learner_name: str = "learner"
    max_steps: Optional[int] = None
    reward_factory: Optional[Callable[[], Callable]] = None
    transform_factory: Optional[Callable[[], Callable]] = None

    def build(self):
        from .fast_env import BomberEnv

        reward_fn = self.reward_factory() if self.reward_factory is not None else None
        env = BomberEnv(
            opponents=self.opponents,
            scenario=self.scenario,
            seed=self.seed,
            learner_name=self.learner_name,
            reward_fn=reward_fn,
            max_steps=self.max_steps,
        )
        transform = self.transform_factory() if self.transform_factory is not None else None
        return env, transform


def _worker(remote, parent_remote, spec: EnvSpec):
    """Child process loop: own one environment, answer commands."""
    parent_remote.close()
    env, transform = spec.build()

    def encode(state):
        if state is None:
            return None
        return transform(state) if transform is not None else state

    try:
        while True:
            command, payload = remote.recv()

            if command == CMD_RESET:
                remote.send(encode(env.reset()))

            elif command == CMD_STEP:
                state, reward, done, info = env.step(payload)
                observation = encode(state)
                if done:
                    # Keep the true final observation for bootstrapping, then
                    # hand back the first observation of the next episode.
                    info = dict(info)
                    info["terminal_observation"] = observation
                    info["round_statistics"] = env.round_statistics()
                    observation = encode(env.reset())
                remote.send((observation, reward, done, info))

            elif command == CMD_STATS:
                remote.send(env.round_statistics())

            elif command == CMD_CLOSE:
                env.close()
                remote.close()
                break

            else:  # pragma: no cover
                raise RuntimeError(f"Unknown command {command!r}")
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


class SubprocVecEnv:
    """A batch of environments, each in its own process."""

    def __init__(self, specs: Sequence[EnvSpec], start_method: str = "spawn"):
        self.n_envs = len(specs)
        if self.n_envs == 0:
            raise ValueError("SubprocVecEnv needs at least one EnvSpec")

        context = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[context.Pipe() for _ in range(self.n_envs)])
        self.processes = []

        for work_remote, remote, spec in zip(self.work_remotes, self.remotes, specs):
            process = context.Process(
                target=_worker, args=(work_remote, remote, spec), daemon=True
            )
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.closed = False

    # -- API ----------------------------------------------------------------
    def reset(self) -> List[Any]:
        for remote in self.remotes:
            remote.send((CMD_RESET, None))
        return [remote.recv() for remote in self.remotes]

    def step(self, actions: Sequence[str]) -> Tuple[List[Any], np.ndarray, np.ndarray, List[dict]]:
        if len(actions) != self.n_envs:
            raise ValueError(f"expected {self.n_envs} actions, got {len(actions)}")

        # Send every command before reading any reply, so the workers run
        # concurrently instead of being serialised by the round-trip.
        for remote, action in zip(self.remotes, actions):
            remote.send((CMD_STEP, action))
        results = [remote.recv() for remote in self.remotes]

        observations, rewards, dones, infos = zip(*results)
        return (
            list(observations),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(infos),
        )

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send((CMD_CLOSE, None))
            except (BrokenPipeError, OSError):  # pragma: no cover
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():  # pragma: no cover
                process.terminate()
        self.closed = True

    def __enter__(self) -> "SubprocVecEnv":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __len__(self) -> int:
        return self.n_envs


class DummyVecEnv:
    """Same interface, no processes -- for debugging and small runs.

    Worth keeping: a crash inside a child process surfaces as an opaque pipe
    error, whereas the same bug here raises with a usable traceback.
    """

    def __init__(self, specs: Sequence[EnvSpec]):
        self.n_envs = len(specs)
        self.envs = []
        self.transforms = []
        for spec in specs:
            env, transform = spec.build()
            self.envs.append(env)
            self.transforms.append(transform)
        self.closed = False

    def _encode(self, index: int, state):
        if state is None:
            return None
        transform = self.transforms[index]
        return transform(state) if transform is not None else state

    def reset(self) -> List[Any]:
        return [self._encode(i, env.reset()) for i, env in enumerate(self.envs)]

    def step(self, actions: Sequence[str]):
        # zip() would silently truncate a short action list and drop the
        # remaining environments' transitions without any error.
        if len(actions) != self.n_envs:
            raise ValueError(f"expected {self.n_envs} actions, got {len(actions)}")

        observations, rewards, dones, infos = [], [], [], []
        for index, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, done, info = env.step(action)
            observation = self._encode(index, state)
            if done:
                info = dict(info)
                info["terminal_observation"] = observation
                info["round_statistics"] = env.round_statistics()
                observation = self._encode(index, env.reset())
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (
            observations,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()
        self.closed = True

    def __enter__(self) -> "DummyVecEnv":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __len__(self) -> int:
        return self.n_envs


def make_specs(
    n_envs: int,
    base_seed: int,
    opponents: Sequence[str] = (),
    scenario: str = "classic",
    reward_factory: Optional[Callable] = None,
    transform_factory: Optional[Callable] = None,
) -> List[EnvSpec]:
    """Build ``n_envs`` specs with decorrelated but reproducible seeds.

    Distinct seeds matter: identical seeds would have every worker play the same
    arena, turning a batch of N transitions into N copies of one.
    """
    return [
        EnvSpec(
            opponents=tuple(opponents),
            scenario=scenario,
            seed=worker_seed(base_seed, index),
            reward_factory=reward_factory,
            transform_factory=transform_factory,
        )
        for index in range(n_envs)
    ]

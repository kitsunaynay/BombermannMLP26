"""In-process wrappers for the provided baseline agents."""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional

from .fast_env import null_logger

#: Baselines shipped with the framework, weakest first.
BASELINE_AGENTS = (
    "random_agent",
    "peaceful_agent",
    "coin_collector_agent",
    "rule_based_agent",
)

#: Environment variable through which each of our agents reads its model path.
MODEL_FILE_VARIABLES = {
    "attackontensor_ppo": "AOT_PPO_MODEL_FILE",
    "attackontensor_ql": "AOT_QL_MODEL_FILE",
}


@dataclass(frozen=True)
class OpponentSpec:
    """One seat in the opponent pool, parsed from a string.

    ``rule_based_agent``
        the agent as shipped.
    ``rule_based_agent:nobomb``
        same policy, ``BOMB`` replaced by ``WAIT``. The "no-bomb intermediate
        stage" both Skynet and Meisheri et al. used: keeps the pursuit
        pressure and the body-blocking, removes the false-positive kills an
        opponent hands out by blowing itself up.
    ``attackontensor_ppo@/abs/path/policy.pt``
        one of our agents playing a *frozen* checkpoint, so self-play against
        past snapshots never touches the ``policy.pt`` the tournament loads.

    Modifiers combine: ``attackontensor_ppo@/x.pt:nobomb``.
    """

    code_name: str
    model_file: Optional[str] = None
    no_bomb: bool = False

    @classmethod
    def parse(cls, spec: str) -> "OpponentSpec":
        text = spec.strip()
        no_bomb = False
        if text.endswith(":nobomb"):
            text, no_bomb = text[: -len(":nobomb")], True
        code_name, model_file = text, None
        if "@" in text:
            code_name, model_file = text.split("@", 1)
            model_file = str(Path(model_file).expanduser().resolve())
        if not code_name:
            raise ValueError(f"empty agent name in opponent spec {spec!r}")
        return cls(code_name=code_name, model_file=model_file, no_bomb=no_bomb)

    @property
    def display_name(self) -> str:
        """Short, filesystem-safe name for logs and result tables."""
        name = self.code_name
        if self.model_file:
            name += "@" + Path(self.model_file).stem
        if self.no_bomb:
            name += "-nobomb"
        return name


def display_name(spec: str) -> str:
    return OpponentSpec.parse(spec).display_name


@contextmanager
def _model_file_override(spec: OpponentSpec) -> Iterator[None]:
    """Point one of our agents at a frozen checkpoint while its ``setup`` runs.

    Our agents read their weights path from the environment (``config.py``) and
    only inside ``setup``, so the override needs to be live just for that call.
    It is restored afterwards so a learner in the same process, or a training
    worker, keeps its own path.
    """
    variable = MODEL_FILE_VARIABLES.get(spec.code_name)
    if spec.model_file is None:
        yield
        return
    if variable is None:
        raise ValueError(f"{spec.code_name} does not take a model file")
    if not Path(spec.model_file).is_file():
        raise FileNotFoundError(f"frozen checkpoint not found: {spec.model_file}")
    previous = os.environ.get(variable)
    os.environ[variable] = spec.model_file
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


class ScriptedOpponent:
    """A provided agent driven directly, without a backend or log files.

    ``code_name`` may be any :class:`OpponentSpec` string.
    """

    def __init__(self, code_name: str, logger=None):
        self.spec = OpponentSpec.parse(code_name)
        self.code_name = self.spec.code_name
        self.module = importlib.import_module(f"agent_code.{self.code_name}.callbacks")

        self.state = SimpleNamespace(
            train=False,
            logger=logger if logger is not None else null_logger(f"blib.opponent.{self.code_name}"),
        )
        with _model_file_override(self.spec):
            self.module.setup(self.state)

    @staticmethod
    def parse_name(spec: str) -> str:
        return display_name(spec)

    def act(self, game_state: dict) -> str:
        """Choose an action, defaulting to ``WAIT``.

        ``rule_based_agent.act`` falls off the end of its loop and returns
        ``None`` when no proposal survives its validity filter, which the real
        framework would turn into an INVALID_ACTION. ``WAIT`` is the same thing
        the framework substitutes elsewhere and keeps the type contract clean.
        """
        action = self.module.act(self.state, game_state)
        if self.spec.no_bomb and action == "BOMB":
            return "WAIT"
        return action if action else "WAIT"

    def reset(self) -> None:
        """Reset per-round memory between episodes.

        ``rule_based_agent`` keeps a bomb history and a coordinate history and
        clears them itself when it notices the round number changed, so nothing
        is required here; agents that expose a ``reset_self`` get it called.
        """
        reset = getattr(self.module, "reset_self", None)
        if callable(reset):
            reset(self.state)


def make_opponents(code_names, logger=None):
    return [ScriptedOpponent(name, logger=logger) for name in code_names]

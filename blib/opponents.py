"""In-process wrappers for the provided baseline agents.

The framework normally reaches an agent through ``AgentRunner``, which imports
its module, builds a ``SimpleNamespace`` to act as ``self``, and attaches a file
logger (agents.py:194). For fast training we want the policy without the
plumbing, so this reproduces the minimum: import the module, hand it a namespace
with ``train`` and ``logger``, and call ``act`` directly.

The baseline agents' code is used *unmodified*. That matters -- ``rule_based_agent``
is the benchmark the brief says you must beat, so training against a subtly
different reimplementation of it would invalidate the comparison.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Optional

from .fast_env import null_logger

#: Baselines shipped with the framework, weakest first.
BASELINE_AGENTS = (
    "random_agent",
    "peaceful_agent",
    "coin_collector_agent",
    "rule_based_agent",
)


class ScriptedOpponent:
    """A provided agent driven directly, without a backend or log files."""

    def __init__(self, code_name: str, logger=None):
        self.code_name = code_name
        self.module = importlib.import_module(f"agent_code.{code_name}.callbacks")

        self.state = SimpleNamespace(
            train=False,
            logger=logger if logger is not None else null_logger(f"blib.opponent.{code_name}"),
        )
        self.module.setup(self.state)

    def act(self, game_state: dict) -> str:
        """Choose an action, defaulting to ``WAIT``.

        ``rule_based_agent.act`` falls off the end of its loop and returns
        ``None`` when no proposal survives its validity filter, which the real
        framework would turn into an INVALID_ACTION. ``WAIT`` is the same thing
        the framework substitutes elsewhere and keeps the type contract clean.
        """
        action = self.module.act(self.state, game_state)
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

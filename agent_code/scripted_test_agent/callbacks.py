"""Deterministic agent used only as a test fixture.

Not a competitor and not a submission. It exists so ``tests/test_fast_env_parity.py``
can drive a stock ``BombeRLeWorld`` and ``blib.fast_env.FastWorld`` through the
*same* action sequence and assert the two worlds evolve identically.

The action is a pure function of ``(round, step, agent name)`` with no random
state at all, so both worlds can generate it independently and still agree.
"""

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def script_action(round_number: int, step: int, name: str) -> str:
    """Pure, reproducible action choice -- no RNG, no hidden state."""
    mixed = round_number * 1_000_003 + step * 31 + sum(ord(c) for c in name)
    return ACTIONS[mixed % len(ACTIONS)]


def setup(self):
    self.logger.debug("scripted_test_agent ready")


def act(self, game_state: dict) -> str:
    name = game_state["self"][0]
    return script_action(game_state["round"], game_state["step"], name)

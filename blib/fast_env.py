"""In-process game environment for fast, parallel training."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import events as e
import settings as s
from agents import Agent
from environment import BombeRLeWorld

ActionProvider = Callable[[Dict[str, dict]], Dict[str, str]]

#: Fields of the argparse namespace that the world touches. Collected here so a
#: change upstream surfaces as a clear AttributeError rather than a mystery.
DEFAULT_ARGS = dict(
    no_gui=True,
    fps=15,
    turn_based=False,
    update_interval=0.1,
    save_replay=False,
    replay=None,
    make_video=False,
    continue_without_training=True,
    log_dir="logs",
    save_stats=False,
    match_name=None,
    seed=None,
    silence_errors=False,
    scenario="classic",
)


def world_args(**overrides) -> SimpleNamespace:
    """Build the argument namespace the world expects."""
    values = dict(DEFAULT_ARGS)
    values.update(overrides)
    return SimpleNamespace(**values)


def null_logger(name: str = "blib.null") -> logging.Logger:
    """A logger that discards everything.

    The framework opens ``logs/game.log`` in ``mode="w"`` (environment.py:61) and
    one file per agent (agents.py:228). With dozens of worker processes those
    handles collide and the disk traffic dominates the run.
    """
    logger = logging.getLogger(name)
    logger.handlers = [logging.NullHandler()]
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    return logger


class PuppetAgent(Agent):
    """An :class:`Agent` whose actions are injected rather than computed.

    Subclassing the real thing (instead of faking it) keeps all the bookkeeping
    the world relies on -- events, scores, statistics, ``bombs_left``, death --
    identical to a tournament game. ``setup`` is a no-op and ``backend`` is
    ``None``, mirroring ``ReplayAgent`` (replay.py:79).
    """

    def __init__(self, name: str, colour: str, train: bool = False):
        super().__init__(name, None, name, train, None, colour, colour)

    def setup(self):  # noqa: D102 - deliberately does nothing
        pass


class FastWorld(BombeRLeWorld):
    """A ``BombeRLeWorld`` driven by an action callback."""

    #: ``get_state_for_agent`` reads this (environment.py:408) but only
    #: ``do_step`` assigns it, so observing the board before the first step --
    #: which is exactly what ``BomberEnv.reset`` does -- would raise. There is no
    #: GUI here, so it is always None.
    user_input = None

    def __init__(
        self,
        agent_names: Sequence[str],
        action_provider: ActionProvider,
        scenario: str = "classic",
        seed: Optional[int] = None,
        **arg_overrides,
    ):
        self.action_provider = action_provider
        self._agent_names = list(agent_names)

        args = world_args(scenario=scenario, seed=seed, **arg_overrides)
        super().__init__(args, [(name, False) for name in self._agent_names])

    # -- overrides ----------------------------------------------------------
    def setup_logging(self):
        """Discard logging instead of opening a shared file per world."""
        self.logger = null_logger("blib.fast_env")

    def setup_agents(self, agents):
        """Create puppets directly, skipping backends and module imports."""
        self.agents = []
        for index, (name, _) in enumerate(agents):
            colour = s.AGENT_COLORS[index % len(s.AGENT_COLORS)]
            self.agents.append(PuppetAgent(name, colour))

    def add_agent(self, agent_dir, name, train=False):  # pragma: no cover
        raise NotImplementedError("FastWorld builds its agents in setup_agents")

    def poll_and_run_agents(self):
        """Mirror of the upstream method with the backend round-trip removed.

        The ordering upstream guarantees is preserved exactly: *every* agent
        observes the world before *any* agent moves, and the moves are then
        applied in a random permutation, so two agents targeting one tile race
        the same way they would in a real game (environment.py:420).
        """
        states: Dict[str, dict] = {}
        for agent in self.active_agents:
            state = self.get_state_for_agent(agent)
            agent.store_game_state(state)
            agent.reset_game_events()
            states[agent.name] = state

        actions = self.action_provider(states)

        permutation = self.rng.permutation(len(self.active_agents))
        self.replay["permutations"].append(permutation)

        for index in permutation:
            agent = self.active_agents[index]
            action = actions.get(agent.name, "WAIT")
            agent.last_action = action
            agent.note_stat("steps")
            self.replay["actions"][agent.name].append(action)
            self.perform_agent_action(agent, action)

    def end(self):
        """Skip the stats file the base class would write."""
        if self.running:
            self.end_round()

    # -- convenience --------------------------------------------------------
    def agent_by_name(self, name: str) -> Agent:
        for agent in self.agents:
            if agent.name == name:
                return agent
        raise KeyError(name)


class BomberEnv:
    """Single-learner environment wrapping :class:`FastWorld`.

    One agent is controlled by the caller; the rest are scripted opponents. The
    API is deliberately gym-shaped (``reset`` / ``step``) so a rollout collector
    does not need to know anything about the framework.

    Rewards are computed by a caller-supplied function over the framework's own
    event list, so training here and training through ``train.py`` optimise the
    same objective.
    """

    def __init__(
        self,
        opponents: Sequence[str] = (),
        scenario: str = "classic",
        seed: Optional[int] = None,
        learner_name: str = "learner",
        reward_fn: Optional[Callable] = None,
        max_steps: Optional[int] = None,
    ):
        from .opponents import ScriptedOpponent

        self.learner_name = learner_name
        self.scenario = scenario
        self.max_steps = max_steps or s.MAX_STEPS
        self.reward_fn = reward_fn

        self.opponent_names: List[str] = []
        self.opponents: Dict[str, ScriptedOpponent] = {}
        for index, code_name in enumerate(opponents):
            # Names must be unique even when the same opponent appears twice.
            name = f"{ScriptedOpponent.parse_name(code_name)}_{index}"
            self.opponent_names.append(name)
            self.opponents[name] = ScriptedOpponent(code_name)

        names = [learner_name] + self.opponent_names
        self.world = FastWorld(names, self._provide_actions, scenario=scenario, seed=seed)

        self._pending_action: str = "WAIT"
        self._states: Dict[str, dict] = {}

    # -- action plumbing ----------------------------------------------------
    def _provide_actions(self, states: Dict[str, dict]) -> Dict[str, str]:
        self._states = states
        actions = {self.learner_name: self._pending_action}
        for name, opponent in self.opponents.items():
            state = states.get(name)
            if state is not None:
                actions[name] = opponent.act(state)
        return actions

    # -- gym-ish API --------------------------------------------------------
    @property
    def learner(self) -> Agent:
        return self.world.agent_by_name(self.learner_name)

    def reset(self) -> Optional[dict]:
        if self.world.running:
            self.world.end_round()
        for opponent in self.opponents.values():
            opponent.reset()
        self.world.new_round()
        return self.world.get_state_for_agent(self.learner)

    def step(self, action: str):
        """Advance one step. Returns ``(state, reward, done, info)``.

        ``state`` is ``None`` once the learner is dead, matching what the real
        framework hands to a dead agent's callbacks (environment.py:397).
        """
        learner = self.learner

        self._pending_action = action
        self.world.do_step()

        # Read the pre-action state *after* do_step, not before. do_step
        # increments the step counter and only then polls, and it is that poll
        # which calls store_game_state (environment.py:424). So before do_step
        # `last_game_state` still holds the *previous* step's observation, and
        # using it here would pair every action with a stale state -- exactly
        # the (s, a, s') mismatch that quietly poisons reward shaping. Afterwards
        # it holds the state the agent actually acted on, which is precisely what
        # the framework passes to game_events_occurred.
        old_state = learner.last_game_state

        events = list(learner.events)
        alive = not learner.dead
        new_state = self.world.get_state_for_agent(learner) if alive else None
        done = (not self.world.running) or learner.dead

        if done and alive:
            # end_round() appends SURVIVED_ROUND to the live agents; mirror that
            # here so the reward matches what train.py would have seen.
            if e.SURVIVED_ROUND not in events and not self.world.running:
                events.append(e.SURVIVED_ROUND)

        reward = 0.0
        if self.reward_fn is not None:
            reward = self.reward_fn(old_state, action, new_state, events)

        info = {
            "events": events,
            "score": learner.score,
            "step": self.world.step,
            "alive": alive,
        }
        return new_state, reward, done, info

    def close(self) -> None:
        if self.world.running:
            self.world.end_round()

    # -- diagnostics --------------------------------------------------------
    def round_statistics(self) -> Dict[str, float]:
        learner = self.learner
        stats = dict(learner.statistics)
        stats["score"] = learner.score
        stats["steps"] = self.world.step
        stats["survived"] = int(not learner.dead)
        return stats

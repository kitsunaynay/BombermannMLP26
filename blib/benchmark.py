"""Headless evaluation against the provided baselines.

Two backends, deliberately:

``fast``
    Drives :class:`blib.fast_env.FastWorld` in-process. Quick enough to run
    between training stages, and it can time each ``act`` call to report
    inference latency. Validated against the real world by
    ``tests/test_fast_env_parity.py``.
``main``
    Shells out to ``python main.py play --no-gui --save-stats`` and reads the
    JSON the framework writes. Slower, but it exercises the *exact* code path
    the tournament uses -- including the think-time accounting that replaces a
    slow agent's action with ``WAIT`` and docks its next step
    (environment.py:448). Numbers quoted as final results should come from here.

Every agent is evaluated on the *same* arena seeds, so differences reflect
agents rather than luck. Aggregates carry bootstrap confidence intervals.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import settings as s

from .fast_env import FastWorld
from .metrics import EpisodeStats, aggregate, mark_winners, stats_from_agent
from .opponents import ScriptedOpponent, display_name
from .seeding import derive_seed, evaluation_seeds, seed_everything

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class MatchConfig:
    agents: Sequence[str]
    scenario: str = "classic"
    n_seeds: int = 20
    rounds_per_seed: int = 1
    base_seed: int = 20260921
    track_latency: bool = True

    @property
    def seeds(self) -> List[int]:
        return evaluation_seeds(self.base_seed, self.n_seeds)


# --------------------------------------------------------------------------
# Fast backend
# --------------------------------------------------------------------------


def play_rounds_fast(
    agents: Sequence[str],
    scenario: str,
    seed: int,
    n_rounds: int = 1,
    track_latency: bool = True,
) -> List[EpisodeStats]:
    """Play ``n_rounds`` in-process and return per-agent stats for each round.

    All agents -- ours and the baselines alike -- are driven through
    :class:`ScriptedOpponent`, which is the same import-and-call path
    ``AgentRunner`` uses. One code path for everyone keeps the comparison fair.
    """
    # Unique display names, so the same agent can appear more than once.
    names = [f"{display_name(code)}_{index}" for index, code in enumerate(agents)]
    # Our own agents build a private `default_rng(config.seed)` in `setup`, and
    # both default to unseeded, so their tie-breaking was a second source of
    # run-to-run drift that seeding the global generators cannot reach. Pin it
    # from the arena seed before `setup` runs, unless the caller has already
    # chosen a seed (training does, and those must not be overwritten).
    for prefix in ("AOT_QL_SEED", "AOT_PPO_SEED"):
        os.environ.setdefault(prefix, str(derive_seed(seed, "agent")))
    policies = {name: ScriptedOpponent(code) for name, code in zip(names, agents)}
    latencies: Dict[str, List[float]] = {name: [] for name in names}

    def provide(states: Dict[str, dict]) -> Dict[str, str]:
        actions = {}
        for name, state in states.items():
            if track_latency:
                started = time.perf_counter()
                actions[name] = policies[name].act(state)
                latencies[name].append((time.perf_counter() - started) * 1000.0)
            else:
                actions[name] = policies[name].act(state)
        return actions

    world = FastWorld(names, provide, scenario=scenario, seed=seed)

    collected: List[EpisodeStats] = []
    for round_index in range(n_rounds):
        for policy in policies.values():
            policy.reset()
        # The arena is reproducible -- `BombeRLeWorld` draws it from its own
        # `default_rng(seed)` (environment.py:335) -- but the *opponents* were
        # not. `rule_based_agent.callbacks.setup` calls a bare
        # `np.random.seed()` (line 69), reseeding the global generator from OS
        # entropy, and the policy then shuffles its action ideas with the
        # unseeded `random.shuffle`. Seven runs of one fixed table over these
        # same 100 arenas scored 3.33 to 3.92 (sd 0.21) because of it, which is
        # wider than most differences this project has tried to measure.
        #
        # Seeding here, after `setup` has done its damage and before the round
        # starts, makes a benchmark reproducible. Per round rather than once,
        # so a round's outcome does not depend on how many rounds preceded it.
        seed_everything(derive_seed(seed, "opponents", round_index))
        for name in names:
            latencies[name] = []

        world.new_round()
        while world.running and world.step < s.MAX_STEPS:
            world.do_step()
        if world.running:
            world.end_round()

        round_stats = [
            stats_from_agent(
                agent,
                seed=seed,
                round_index=round_index,
                latencies=latencies.get(agent.name),
            )
            for agent in world.agents
        ]
        mark_winners(round_stats)
        collected.extend(round_stats)

    return collected


def benchmark_fast(config: MatchConfig, progress: bool = False) -> Dict[str, object]:
    """Run the whole seed sweep with the in-process backend."""
    all_stats: List[EpisodeStats] = []

    for index, seed in enumerate(config.seeds):
        if progress:
            print(f"  seed {index + 1}/{config.n_seeds} ({seed})", flush=True)
        all_stats.extend(
            play_rounds_fast(
                config.agents,
                config.scenario,
                seed,
                n_rounds=config.rounds_per_seed,
                track_latency=config.track_latency,
            )
        )

    return summarise(all_stats, config, backend="fast")


# --------------------------------------------------------------------------
# main.py backend -- the certified path
# --------------------------------------------------------------------------


def benchmark_main(
    config: MatchConfig,
    python: Optional[str] = None,
    progress: bool = False,
) -> Dict[str, object]:
    """Run each seed as a real ``main.py`` subprocess and parse its stats JSON.

    Slower by far, but this is the code path the graders run, so it is what
    final numbers should be quoted from.
    """
    python = python or sys.executable
    all_stats: List[EpisodeStats] = []

    results_dir = REPO_ROOT / "results" / "_benchmark_tmp"
    results_dir.mkdir(parents=True, exist_ok=True)

    for index, seed in enumerate(config.seeds):
        if progress:
            print(f"  seed {index + 1}/{config.n_seeds} ({seed})", flush=True)

        output = results_dir / f"seed_{seed}.json"
        command = [
            python,
            "main.py",
            "play",
            "--no-gui",
            "--n-rounds",
            str(config.rounds_per_seed),
            "--scenario",
            config.scenario,
            "--seed",
            str(seed),
            "--agents",
            *config.agents,
            "--save-stats",
            str(output),
        ]

        completed = subprocess.run(
            command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"main.py failed for seed {seed}:\n{completed.stderr[-2000:]}"
            )

        all_stats.extend(_stats_from_main_json(json.loads(output.read_text()), seed))

    return summarise(all_stats, config, backend="main")


def _stats_from_main_json(payload: dict, seed: int) -> List[EpisodeStats]:
    """Convert the framework's results JSON into EpisodeStats.

    ``main.py`` reports *lifetime* totals per agent plus per-round aggregates
    (environment.py:311), so these are whole-run figures rather than per-round
    ones; ``rounds`` normalises them.

    Two metrics cannot be recovered from that JSON and must not be read as
    measurements:

    * **survival** -- there is no per-agent survival field, and being killed by
      an opponent is indistinguishable from surviving in the totals;
    * **win rate** -- ``mark_winners`` here compares whole-run mean scores, so it
      means "won the match", not "won the round" as in the fast backend.

    :func:`format_table` prints ``-`` for survival on this backend rather than a
    misleading zero.
    """
    collected: List[EpisodeStats] = []
    by_agent = payload.get("by_agent", {})
    rounds = max(1, len(payload.get("by_round", {})))

    for name, values in by_agent.items():
        collected.append(
            EpisodeStats(
                agent=name,
                seed=seed,
                round_index=0,
                score=float(values.get("score", 0)) / rounds,
                coins=int(values.get("coins", 0)) // rounds,
                crates=int(values.get("crates", 0)) // rounds,
                kills=int(values.get("kills", 0)) // rounds,
                suicides=int(values.get("suicides", 0)) // rounds,
                invalid=int(values.get("invalid", 0)) // rounds,
                moves=int(values.get("moves", 0)) // rounds,
                bombs=int(values.get("bombs", 0)) // rounds,
                steps=max(1, int(values.get("steps", 0)) // rounds),
            )
        )

    mark_winners(collected)
    return collected


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def summarise(
    stats: Sequence[EpisodeStats],
    config: MatchConfig,
    backend: str,
) -> Dict[str, object]:
    """Group per-round stats by agent and aggregate each group."""
    by_agent: Dict[str, List[EpisodeStats]] = {}
    for stat in stats:
        by_agent.setdefault(stat.agent, []).append(stat)

    return {
        "backend": backend,
        "scenario": config.scenario,
        "agents": list(config.agents),
        "n_seeds": config.n_seeds,
        "rounds_per_seed": config.rounds_per_seed,
        "base_seed": config.base_seed,
        "per_agent": {name: aggregate(group) for name, group in by_agent.items()},
        "rows": [stat.as_row() for stat in stats],
    }


def format_table(summary: Dict[str, object]) -> str:
    """Fixed-width comparison table for the terminal."""
    per_agent = summary["per_agent"]
    header = (
        f"{'agent':<28} {'score':>14} {'win%':>7} {'surv%':>7} "
        f"{'coins':>7} {'crates':>7} {'kills':>6} {'suic%':>7} {'p95 ms':>8}"
    )
    lines = [
        f"scenario={summary['scenario']}  backend={summary['backend']}  "
        f"seeds={summary['n_seeds']}x{summary['rounds_per_seed']} rounds",
        header,
        "-" * len(header),
    ]

    for name, values in sorted(
        per_agent.items(), key=lambda item: -item[1].get("score_mean", 0.0)
    ):
        score = values.get("score_mean", 0.0)
        low = values.get("score_ci_low", score)
        high = values.get("score_ci_high", score)

        latency = values.get("latency_p95_ms")
        latency_cell = f"{latency:>8.2f}" if latency is not None else f"{'-':>8}"

        # The main.py results JSON carries no per-agent survival field, so a
        # zero there would be an artefact rather than a measurement.
        if summary["backend"] == "fast":
            survival_cell = f"{100 * values.get('survival_rate', 0):>6.1f}"
        else:
            survival_cell = f"{'-':>6}"

        lines.append(
            f"{name:<28} "
            f"{score:>6.2f}[{low:4.1f},{high:4.1f}] "
            f"{100 * values.get('win_rate', 0):>6.1f} "
            f"{survival_cell} "
            f"{values.get('coins_mean', 0):>7.2f} "
            f"{values.get('crates_mean', 0):>7.2f} "
            f"{values.get('kills_mean', 0):>6.2f} "
            f"{100 * values.get('suicide_rate', 0):>6.1f} "
            f"{latency_cell}"
        )

    return "\n".join(lines)

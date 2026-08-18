"""Per-episode statistics and aggregation.

The brief makes performance metrics a graded item -- *"defining good performance
metrics ... [is] crucial for a good grade"* -- and asks that results be compared
systematically rather than anecdotally. So aggregates come with bootstrap
confidence intervals: with 400-step episodes on randomly generated arenas the
round-to-round variance is large, and a bare mean over ten rounds cannot
distinguish a real improvement from noise.

Metrics tracked, and why each one earns its place:

``score``               the tournament objective; everything else is diagnosis
``coins`` / ``coin_rate``   Task 1-2 progress
``crates``              whether the agent uses bombs productively at all
``kills`` / ``suicides``    Task 3-4 progress; suicides are the classic failure
``steps`` / ``survival_rate``   survival, which dominates tournament scoring
``invalid_rate``        walking into walls: a cheap sanity check on the policy
``latency_*``           the 0.5 s budget is a hard constraint, not a nicety
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

import events as e
import settings as s


@dataclass
class EpisodeStats:
    """Outcome of one round for one agent."""

    agent: str = ""
    seed: Optional[int] = None
    round_index: int = 0

    score: float = 0.0
    coins: int = 0
    crates: int = 0
    kills: int = 0
    suicides: int = 0
    invalid: int = 0
    moves: int = 0
    bombs: int = 0
    steps: int = 0
    survived: int = 0
    won: int = 0
    total_reward: float = 0.0

    latencies_ms: List[float] = field(default_factory=list)

    # -- derived ------------------------------------------------------------
    @property
    def invalid_rate(self) -> float:
        return self.invalid / self.steps if self.steps else 0.0

    @property
    def coins_per_step(self) -> float:
        return self.coins / self.steps if self.steps else 0.0

    def as_row(self) -> Dict[str, float]:
        row = {k: v for k, v in asdict(self).items() if k != "latencies_ms"}
        row["invalid_rate"] = self.invalid_rate
        row["coins_per_step"] = self.coins_per_step
        if self.latencies_ms:
            values = np.asarray(self.latencies_ms)
            row["latency_mean_ms"] = float(values.mean())
            row["latency_p95_ms"] = float(np.percentile(values, 95))
            row["latency_max_ms"] = float(values.max())
        return row


def stats_from_agent(agent, seed=None, round_index=0, latencies=None) -> EpisodeStats:
    """Build stats from a framework ``Agent`` after a round.

    Reads the counters ``Agent.note_stat`` maintains (agents.py:30) rather than
    recounting events, so the numbers match the framework's own bookkeeping.
    """
    counters = agent.statistics
    return EpisodeStats(
        agent=agent.name,
        seed=seed,
        round_index=round_index,
        score=float(agent.score),
        coins=int(counters.get("coins", 0)),
        crates=int(counters.get("crates", 0)),
        kills=int(counters.get("kills", 0)),
        suicides=int(counters.get("suicides", 0)),
        invalid=int(counters.get("invalid", 0)),
        moves=int(counters.get("moves", 0)),
        bombs=int(counters.get("bombs", 0)),
        steps=int(counters.get("steps", 0)),
        survived=int(not agent.dead),
        latencies_ms=list(latencies or []),
    )


def mark_winners(stats: Sequence[EpisodeStats]) -> None:
    """Flag the highest-scoring agent(s) of a round.

    Ties are recorded as wins for everyone tied, which keeps the win rates of a
    round summing to at least one and avoids inventing a tie-break the
    tournament does not define.
    """
    if not stats:
        return
    best = max(stat.score for stat in stats)
    for stat in stats:
        stat.won = int(stat.score == best)


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple:
    """Percentile bootstrap confidence interval for the mean.

    Non-parametric on purpose: round scores are small integers with a heavy
    spike at zero, so a normal-theory interval would misstate the uncertainty.
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return (float("nan"), float("nan"))
    if array.size == 1:
        return (float(array[0]), float(array[0]))

    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(resamples, array.size), replace=True)
    means = draws.mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.percentile(means, 100 * tail)),
        float(np.percentile(means, 100 * (1.0 - tail))),
    )


AGGREGATED_FIELDS = (
    "score",
    "coins",
    "crates",
    "kills",
    "suicides",
    "steps",
    "survived",
    "won",
    "invalid_rate",
    "total_reward",
)


def aggregate(stats: Sequence[EpisodeStats], confidence: float = 0.95) -> Dict[str, float]:
    """Mean, standard deviation and bootstrap CI for each tracked field."""
    if not stats:
        return {"n_rounds": 0}

    summary: Dict[str, float] = {"n_rounds": len(stats)}

    for name in AGGREGATED_FIELDS:
        values = [float(getattr(stat, name, 0.0)) for stat in stats]
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        low, high = bootstrap_ci(values, confidence=confidence)
        summary[f"{name}_ci_low"] = low
        summary[f"{name}_ci_high"] = high

    # Rates that read more naturally as percentages of rounds.
    summary["win_rate"] = summary["won_mean"]
    summary["survival_rate"] = summary["survived_mean"]
    summary["suicide_rate"] = summary["suicides_mean"]

    latencies = [value for stat in stats for value in stat.latencies_ms]
    if latencies:
        array = np.asarray(latencies)
        summary["latency_mean_ms"] = float(array.mean())
        summary["latency_p50_ms"] = float(np.percentile(array, 50))
        summary["latency_p95_ms"] = float(np.percentile(array, 95))
        summary["latency_max_ms"] = float(array.max())
        summary["latency_budget_ms"] = float(s.TIMEOUT * 1000)
        summary["latency_headroom"] = float(s.TIMEOUT * 1000 / max(array.max(), 1e-9))

    return summary


def coin_efficiency(stats: Sequence[EpisodeStats], coins_available: Optional[int] = None) -> float:
    """Fraction of the coins on the board that were collected.

    Defaults to the scenario's coin count so Task-1 progress reads as a
    proportion rather than a raw count.
    """
    if not stats:
        return 0.0
    if coins_available is None:
        coins_available = s.SCENARIOS["coin-heaven"]["COIN_COUNT"]
    collected = float(np.mean([stat.coins for stat in stats]))
    return collected / coins_available if coins_available else 0.0

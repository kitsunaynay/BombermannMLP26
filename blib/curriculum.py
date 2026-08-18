"""Tasks 1-4 from the brief, expressed as data.

Section 4 of the project brief defines four nested subgoals and says the tasks
"are subsets of each other, so an agent that can handle task 4 should also be
able to solve 1, 2 and 3". Encoding them as data rather than as prose in a
README means the training scripts, the benchmark harness and the report all read
the same definition, and the promotion gates become a measurement rather than a
judgement call.

Each stage carries a ``gate``: the condition the agent must meet before moving
on. The gates double as the plan's failure detectors -- a stage that stops
improving without passing its gate is the signal to change the design, which is
exactly the systematic loop the brief grades on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class Gate:
    """Promotion criteria, evaluated against an aggregated metrics dict."""

    description: str
    conditions: Dict[str, Tuple[str, float]] = field(default_factory=dict)

    def evaluate(self, summary: Dict[str, float]) -> Tuple[bool, Dict[str, bool]]:
        """Return ``(passed, per-condition results)``."""
        results: Dict[str, bool] = {}
        for metric, (operator, threshold) in self.conditions.items():
            value = summary.get(metric)
            if value is None:
                results[metric] = False
                continue
            if operator == ">=":
                results[metric] = value >= threshold
            elif operator == "<=":
                results[metric] = value <= threshold
            elif operator == ">":
                results[metric] = value > threshold
            elif operator == "<":
                results[metric] = value < threshold
            else:  # pragma: no cover
                raise ValueError(f"Unknown operator {operator!r}")
        return (all(results.values()) if results else False), results


@dataclass
class Stage:
    """One curriculum stage."""

    index: int
    name: str
    description: str
    scenario: str
    opponents: Sequence[str]
    gate: Gate
    episodes: int = 2000
    # Q-learning feature set. Every stage uses the SAME variant on purpose:
    # the variant determines what a Q-table key *is*, so a stage that changed it
    # could not reuse the previous stage's table at all -- chaining would
    # silently throw the earlier learning away. `compact` remains available via
    # `tools/train_ql.py --variant compact` as a faster single-stage ablation.
    variant: str = "full"
    notes: str = ""

    @property
    def label(self) -> str:
        return f"task{self.index}-{self.name}"


#: The four tasks, in the order the brief defines them.
STAGES: List[Stage] = [
    Stage(
        index=1,
        name="coin-heaven",
        description=(
            "Collect revealed coins on an empty board. No crates, no opponents, "
            "no bombs required. Tests navigation only."
        ),
        scenario="coin-heaven",
        opponents=(),
        episodes=2000,
        gate=Gate(
            description="collects nearly every coin without walking into walls",
            conditions={
                "coins_mean": (">=", 45.0),
                "invalid_rate_mean": ("<=", 0.02),
                "suicide_rate": ("<=", 0.05),
            },
        ),
        notes=(
            "Bombs are never useful here, so any suicide is pure loss. The "
            "`compact` variant solves this stage faster but cannot chain into "
            "stage 2, so the curriculum keeps `full` throughout."
        ),
    ),
    Stage(
        index=2,
        name="loot-crate",
        description=(
            "Crates but no opponents. The agent must bomb crates to uncover coins "
            "and, crucially, survive its own blasts."
        ),
        scenario="loot-crate",
        opponents=(),
        episodes=4000,
        variant="full",
        gate=Gate(
            description="destroys crates and stops killing itself",
            conditions={
                "suicide_rate": ("<=", 0.05),
                "crates_mean": (">=", 8.0),
                "survival_rate": (">=", 0.8),
            },
        ),
        notes=(
            "The brief calls escaping bombs 'a crucial capability for good "
            "tournament performance', so this stage gets the most episodes."
        ),
    ),
    Stage(
        index=3,
        name="hunt",
        description=(
            "Crates plus the two weak provided agents. peaceful_agent never bombs; "
            "coin_collector_agent bombs only to reach coins."
        ),
        scenario="classic",
        opponents=("peaceful_agent", "coin_collector_agent"),
        episodes=6000,
        variant="full",
        gate=Gate(
            description="beats the weak baselines more often than not",
            conditions={
                "win_rate": (">=", 0.6),
                "suicide_rate": ("<=", 0.1),
            },
        ),
    ),
    Stage(
        index=4,
        name="tournament",
        description=(
            "Full classic game against rule_based_agent, the benchmark the brief "
            "says you must beat to have any chance in the tournament."
        ),
        scenario="classic",
        opponents=("rule_based_agent", "rule_based_agent", "rule_based_agent"),
        episodes=10000,
        variant="full",
        gate=Gate(
            description="beats rule_based_agent",
            conditions={
                "win_rate": (">=", 0.5),
                "score_mean": (">=", 5.0),
            },
        ),
        notes="Self-play variants can replace one or more opponents here.",
    ),
]

STAGES_BY_INDEX: Dict[int, Stage] = {stage.index: stage for stage in STAGES}


def get_stage(index: int) -> Stage:
    try:
        return STAGES_BY_INDEX[index]
    except KeyError:
        raise ValueError(
            f"Unknown curriculum stage {index}; choose from {sorted(STAGES_BY_INDEX)}"
        ) from None


def describe_stage(stage: Stage) -> str:
    opponents = ", ".join(stage.opponents) if stage.opponents else "none (solo)"
    lines = [
        f"Task {stage.index}: {stage.name}",
        f"  {stage.description}",
        f"  scenario  : {stage.scenario}",
        f"  opponents : {opponents}",
        f"  episodes  : {stage.episodes}",
        f"  gate      : {stage.gate.description}",
    ]
    for metric, (operator, threshold) in stage.gate.conditions.items():
        lines.append(f"              {metric} {operator} {threshold}")
    if stage.notes:
        lines.append(f"  note      : {stage.notes}")
    return "\n".join(lines)


def report_gate(stage: Stage, summary: Dict[str, float]) -> str:
    """Human-readable pass/fail table for a stage's gate."""
    passed, results = stage.gate.evaluate(summary)
    lines = [f"Task {stage.index} ({stage.name}) gate: {'PASS' if passed else 'FAIL'}"]
    for metric, (operator, threshold) in stage.gate.conditions.items():
        value = summary.get(metric)
        mark = "ok " if results.get(metric) else "FAIL"
        shown = "n/a" if value is None else f"{value:.4g}"
        lines.append(f"  [{mark}] {metric} {operator} {threshold} (actual {shown})")
    return "\n".join(lines)

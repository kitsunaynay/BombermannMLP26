"""Paired comparison of two benchmark JSONs played on the same arena seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

METRICS = ("score", "coins", "kills", "suicides", "survived", "won", "crates", "steps")
DECISION_THRESHOLD_SCORE = 0.35


def rows_by_seed(summary: dict, agent: str) -> Dict[Tuple[int, int], dict]:
    table = {}
    for row in summary["rows"]:
        if row["agent"] == agent:
            table[(int(row["seed"]), int(row.get("round_index", 0)))] = row
    return table


def guess_agent(summary: dict, hint: str) -> str:
    names = list(summary["per_agent"])
    if hint in names:
        return hint
    matches = [name for name in names if name.startswith(hint)]
    if len(matches) != 1:
        raise SystemExit(f"--agent {hint!r} is ambiguous or absent; agents are {names}")
    return matches[0]


def paired_difference(a: List[float], b: List[float], samples: int = 5000, seed: int = 0):
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(samples)
    ])
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("incumbent", type=Path)
    parser.add_argument("--agent", default="attackontensor_ppo",
                        help="agent name (or unique prefix) to compare, in both files")
    args = parser.parse_args(argv)

    cand = json.loads(args.candidate.read_text())
    inc = json.loads(args.incumbent.read_text())
    name_c = guess_agent(cand, args.agent)
    name_i = guess_agent(inc, args.agent)

    rows_c = rows_by_seed(cand, name_c)
    rows_i = rows_by_seed(inc, name_i)
    shared = sorted(set(rows_c) & set(rows_i))
    if not shared:
        raise SystemExit("no common arena seeds; were both runs made with the same --base-seed?")
    if len(shared) < len(rows_c) or len(shared) < len(rows_i):
        print(f"note: {len(shared)} common arenas of {len(rows_c)} / {len(rows_i)}")

    print(f"{name_c} ({args.candidate.name}) minus {name_i} ({args.incumbent.name}), "
          f"n = {len(shared)} paired arenas\n")
    print(f"{'metric':<10} {'cand':>8} {'inc':>8} {'diff':>8} {'95% CI':>18}  verdict")
    for metric in METRICS:
        a = [float(rows_c[key].get(metric, 0.0)) for key in shared]
        b = [float(rows_i[key].get(metric, 0.0)) for key in shared]
        diff, low, high = paired_difference(a, b)
        excludes_zero = low > 0 or high < 0
        verdict = "CI excludes 0" if excludes_zero else "not distinguishable"
        if metric == "score" and abs(diff) < DECISION_THRESHOLD_SCORE:
            verdict += f" (|diff| < {DECISION_THRESHOLD_SCORE} threshold)"
        print(f"{metric:<10} {np.mean(a):>8.3f} {np.mean(b):>8.3f} {diff:>+8.3f} "
              f"[{low:>+7.3f}, {high:>+7.3f}]  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

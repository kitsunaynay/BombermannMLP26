"""Aggregate a multi-seed ablation sweep into one comparison table.
    python tools/aggregate_seeds.py --prefix abl2 --stage task2-loot-crate
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

AGENT_KEY = "attackontensor_ql_0"
#: `abl2-safety-hard-s1` -> arm "safety-hard", seed 1.
RUN_ID = re.compile(r"^(?P<prefix>[^-]+)-(?P<arm>.+)-s(?P<seed>\d+)$")

METRICS = ("coins_mean", "crates_mean", "survival_rate", "suicide_rate", "latency_p95_ms")


def mean(values):
    return sum(values) / len(values)


def stdev(values):
    """Sample sd. Zero for n < 2 rather than an error: a single-seed arm is
    still worth printing, it just carries no variance estimate."""
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def welch(a, b):
    """Welch's t and its dof for two independent samples; None if either arm has n < 2."""
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = stdev(a) ** 2, stdev(b) ** 2
    na, nb = len(a), len(b)
    se2 = va / na + vb / nb
    if se2 == 0:
        return (0.0, float(na + nb - 2))
    t = (mean(a) - mean(b)) / math.sqrt(se2)
    dof = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return (t, dof)


def two_sided_p(t, dof):
    """Two-sided p for Welch's t, via the regularised incomplete beta."""
    x = dof / (dof + t * t)
    return _betainc(dof / 2.0, 0.5, x)


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b) by continued fraction.

    Hand-rolled so the tool has no scipy dependency: `requirements.txt` keeps
    the submitted agents at numpy/torch, and adding scipy for one p-value would
    be the wrong trade.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _cf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log(1.0 - x) + a * math.log(x)) * _cf(b, a, 1.0 - x) / b


def _cf(a, b, x, iterations=300, tiny=1e-30, eps=1e-14):
    """Lentz's algorithm for the beta continued fraction (Numerical Recipes
    ``betacf``). The two terms of each level have different numerators, so they
    are applied as an explicit pair rather than folded into one loop -- getting
    that wrong silently returns plausible-looking but wrong p-values.
    """
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        # even step
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        # odd step
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


#: Snapshots before this round are undertrained rather than unstable; including
#: them would report 94% of early snapshots as "collapsed" and say nothing.
LATE_FROM_ROUND = 1250


def late_snapshot_stats(selection_path: Path):
    """Mean and sd of the *late* snapshot scores of one run.

    The selected checkpoint measures the best artifact a run produced. It says
    nothing about how reliably the run produced one -- and that is exactly where
    Double Q-learning, custom events and symmetry turned out to differ, while
    being indistinguishable on the selected table. Reporting only the selected
    score hid three real effects.
    """
    if not selection_path.is_file():
        return None
    scores = {r["checkpoint"]: r["score"] for r in json.loads(selection_path.read_text())}
    late = [
        value
        for name, value in scores.items()
        if name.startswith("q_table_r") and int(name[9:15]) >= LATE_FROM_ROUND
    ]
    if not late:
        return None
    mu = mean(late)
    sd = math.sqrt(sum((v - mu) ** 2 for v in late) / len(late))  # population: a full run
    return mu, sd


def collect(prefix: str, stage: str) -> dict:
    """Map arm -> {seed: metrics} from every finished run under ``results/``."""
    arms: dict = defaultdict(dict)
    for path in sorted((REPO_ROOT / "results").glob(f"{prefix}-*/{stage}/benchmark.json")):
        match = RUN_ID.match(path.parent.parent.name)
        if match is None:
            print(f"skipping unparseable run id {path.parent.parent.name}", file=sys.stderr)
            continue
        values = json.loads(path.read_text())["per_agent"][AGENT_KEY]
        selection = path.parent / "checkpoint_selection.json"
        chosen = json.loads(selection.read_text())[0]["checkpoint"] if selection.is_file() else "?"
        record = dict(values, checkpoint=chosen)
        stats = late_snapshot_stats(selection)
        if stats is not None:
            record["late_mean"], record["late_sd"] = stats
        arms[match["arm"]][int(match["seed"])] = record
    return arms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default="abl2")
    parser.add_argument("--stage", default="task2-loot-crate")
    parser.add_argument("--baseline", default="baseline",
                        help="arm every other arm is tested against")
    parser.add_argument("--out", default=None, help="also write the table as JSON")
    args = parser.parse_args()

    arms = collect(args.prefix, args.stage)
    if not arms:
        print(f"no finished runs matching results/{args.prefix}-*/{args.stage}/", file=sys.stderr)
        return 1

    coins = {arm: [seeds[s]["coins_mean"] for s in sorted(seeds)] for arm, seeds in arms.items()}
    reference = coins.get(args.baseline)

    print(f"\n{args.stage}  |  {args.prefix}-*  |  mean +- sd over seeds\n")
    header = (f"{'arm':<22} {'n':>2} {'coins':>16} {'surv':>6} {'suic':>6} "
              f"{'vs baseline':>22} | {'late mean':>9} {'within sd':>9} {'gain':>6}")
    print(header)
    print("-" * len(header))

    for arm in sorted(arms, key=lambda a: -mean(coins[a])):
        seeds = arms[arm]
        series = {m: [seeds[s][m] for s in sorted(seeds)] for m in METRICS}
        n = len(seeds)

        verdict = ""
        if reference is not None and arm != args.baseline:
            test = welch(coins[arm], reference)
            if test is None:
                verdict = "n<2"
            else:
                t, dof = test
                p = two_sided_p(t, dof)
                delta = mean(coins[arm]) - mean(reference)
                mark = "*" if p < 0.05 else " "
                verdict = f"{delta:+6.2f} coins p={p:.3f}{mark}"

        # Policy quality without selection, and what selection bought.
        lm = [seeds[s]["late_mean"] for s in sorted(seeds) if "late_mean" in seeds[s]]
        ls = [seeds[s]["late_sd"] for s in sorted(seeds) if "late_sd" in seeds[s]]
        late = (f"{mean(lm):>9.2f} {mean(ls):>9.2f} "
                f"{mean(series['coins_mean']) - mean(lm):>+6.1f}") if lm else " " * 26

        print(f"{arm:<22} {n:>2} "
              f"{mean(series['coins_mean']):>7.2f} +-{stdev(series['coins_mean']):>5.2f} "
              f"{mean(series['survival_rate']):>5.0%} "
              f"{mean(series['suicide_rate']):>6.2f} "
              f"{verdict:>22} | {late}")

    print("\n* = Welch's t against the baseline arm, p < 0.05. Per-seed coins:")
    for arm in sorted(arms, key=lambda a: -mean(coins[a])):
        per_seed = "  ".join(f"s{s}={arms[arm][s]['coins_mean']:.2f}" for s in sorted(arms[arm]))
        print(f"  {arm:<22} {per_seed}")

    if args.out:
        payload = {
            arm: {
                "per_seed": {str(s): arms[arm][s] for s in sorted(arms[arm])},
                "coins_mean": mean(coins[arm]),
                "coins_sd": stdev(coins[arm]),
                "n_seeds": len(arms[arm]),
            }
            for arm in arms
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

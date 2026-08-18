#!/usr/bin/env python3
"""Verify an agent decides within the tournament's per-step budget.

The brief guarantees "exclusive access to one thread of an AMD Ryzen 5 2600" and
0.5 s per step. Overrunning is not a soft failure: the framework replaces the
action with ``WAIT`` *and* subtracts the overrun from the next step's budget
(environment.py:448), so a slow agent plays a strictly worse game than a fast
one, on top of whatever its policy would have done.

This measures ``act`` exactly as the framework calls it -- one thread, real game
states drawn from a real game, cold start included.

Usage::

    python tools/latency_check.py --agent attackontensor_ppo
    python tools/latency_check.py --agent attackontensor_ql --scenario classic --steps 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import settings as s  # noqa: E402

BUDGET_MS = s.TIMEOUT * 1000.0

#: Alert threshold from the plan: a 5x safety margin under the real budget.
WARN_MS = BUDGET_MS / 5.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True, help="agent_code directory name")
    parser.add_argument("--scenario", default="classic", choices=list(s.SCENARIOS))
    parser.add_argument("--opponents", nargs="*", default=["rule_based_agent"],
                        help="opponents to make the board realistically busy")
    parser.add_argument("--steps", type=int, default=1500, help="act() calls to time")
    parser.add_argument("--threads", type=int, default=1, help="torch threads (tournament: 1)")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import torch

        torch.set_num_threads(max(1, args.threads))
    except ImportError:
        pass  # the Q-learning agent has no torch dependency

    from blib.fast_env import FastWorld
    from blib.opponents import ScriptedOpponent

    names = [f"{args.agent}_0"] + [
        f"{code}_{index + 1}" for index, code in enumerate(args.opponents)
    ]
    codes = [args.agent, *args.opponents]

    print(f"Loading {args.agent} ...")
    cold_started = time.perf_counter()
    policies = {name: ScriptedOpponent(code) for name, code in zip(names, codes)}
    setup_ms = (time.perf_counter() - cold_started) * 1000.0
    print(f"setup() took {setup_ms:.0f} ms (untimed by the framework)\n")

    target = names[0]
    latencies: list = []

    def provide(states):
        actions = {}
        for name, state in states.items():
            if name == target:
                started = time.perf_counter()
                actions[name] = policies[name].act(state)
                latencies.append((time.perf_counter() - started) * 1000.0)
            else:
                actions[name] = policies[name].act(state)
        return actions

    world = FastWorld(names, provide, scenario=args.scenario, seed=args.seed)

    rounds = 0
    while len(latencies) < args.steps:
        world.new_round()
        rounds += 1
        while world.running and world.step < s.MAX_STEPS and len(latencies) < args.steps:
            world.do_step()
        if world.running:
            world.end_round()

    values = np.asarray(latencies)
    if values.size == 0:
        print("No act() calls were timed; the agent died instantly.", file=sys.stderr)
        return 1

    first = values[0]
    print(f"{args.agent} on '{args.scenario}' vs {', '.join(args.opponents) or 'nobody'}")
    print(f"  samples      : {values.size} over {rounds} round(s)")
    print(f"  first call   : {first:8.3f} ms   <- cold start, inside the timed path")
    print(f"  mean         : {values.mean():8.3f} ms")
    print(f"  p50          : {np.percentile(values, 50):8.3f} ms")
    print(f"  p95          : {np.percentile(values, 95):8.3f} ms")
    print(f"  p99          : {np.percentile(values, 99):8.3f} ms")
    print(f"  max          : {values.max():8.3f} ms")
    print(f"  budget       : {BUDGET_MS:8.3f} ms")
    print(f"  headroom     : {BUDGET_MS / values.max():8.1f}x on the worst step")

    over = int((values > BUDGET_MS).sum())
    p95 = float(np.percentile(values, 95))

    print()
    if over:
        print(f"FAIL: {over} step(s) exceeded the {BUDGET_MS:.0f} ms budget.")
        return 1
    if p95 > WARN_MS:
        print(f"WARN: p95 {p95:.1f} ms is above the {WARN_MS:.0f} ms safety threshold.")
        print("      This machine may be faster than the tournament's Ryzen 5 2600.")
        return 1

    print(f"PASS: p95 {p95:.2f} ms is well inside the {BUDGET_MS:.0f} ms budget.")
    print("      Note this machine is likely faster than the tournament CPU; the")
    print("      5x safety threshold is there to absorb that difference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Decide which agent is best
    python tools/competition_test.py
    python tools/competition_test.py --rounds 200 --jobs 32 --label after-ft2
    python tools/competition_test.py --candidates attackontensor_ppo ft2s1=attackontensor_ppo@agent_code/attackontensor_ppo/checkpoints/ppo-ft2-rb3-robust-s1.pt

Results land in ``results/competition/``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tournament_sim as ts  # noqa: E402

REPO_ROOT = ts.REPO_ROOT
OUTPUT_DIR = REPO_ROOT / "results" / "competition"
DEFAULT_CANDIDATES = ["attackontensor_ppo", "attackontensor_ql", "survival_linear_ppo_v4"]
MATCHES = ("baseline", "mixed", "rivals")
MIXED_FIELD = ["rule_based_agent", "coin_collector_agent", "peaceful_agent"]
METRICS = ("score", "coins", "kills", "suicides", "steps", "time", "invalid")
#: The project's decision threshold for a score difference (docs/BLUEPRINT_V2.md, section 4).
SCORE_THRESHOLD = 0.35
SINGLE_THREAD = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    name: str                    # directory name inside the staged framework
    agent_dir: str               # directory under this repo's agent_code/
    checkpoint: Optional[Path]   # model file swapped in, or None for the shipped one


def parse_candidate(spec: str) -> Candidate:
    """``dir``, ``dir@checkpoint`` or ``label=dir@checkpoint``."""
    left, _, checkpoint = spec.partition("@")
    label, _, agent_dir = left.rpartition("=")
    if not agent_dir:
        raise ValueError(f"no agent directory in candidate spec {spec!r}")
    if not checkpoint:
        if label:
            raise ValueError(f"a label needs a checkpoint: {spec!r}")
        return Candidate(agent_dir, agent_dir, None)
    path = Path(checkpoint)
    path = path if path.is_absolute() else REPO_ROOT / path
    label = label or path.stem
    # The framework imports the directory as a module, so keep it an identifier.
    name = f"{agent_dir}__{re.sub(r'[^0-9A-Za-z_]', '_', label)}"
    return Candidate(name, agent_dir, path)


def model_files(directory: Path) -> List[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix in (".pt", ".pkl", ".npz", ".npy"))


def fingerprint(directory: Path) -> Dict[str, str]:
    """Short hashes of the model files: which weights a result file is about."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in model_files(directory)}


def stage_candidate(candidate: Candidate, framework: Path, python: str) -> str:
    """Put the candidate into ``framework``; returns how it got there."""
    base = framework / "agent_code" / candidate.agent_dir
    if not base.exists():
        try:
            ts.stage_submission(candidate.agent_dir, framework, python)
            how = "submission zip"
        except subprocess.CalledProcessError:
            # package_submission.py refused it: copy the directory as it is.
            ts.stage_student(candidate.agent_dir, framework)
            how = "directory copy"
    else:
        how = "already staged"
    if candidate.checkpoint is None:
        return how

    if not candidate.checkpoint.is_file():
        raise SystemExit(f"no such checkpoint: {candidate.checkpoint}")
    target = framework / "agent_code" / candidate.name
    shutil.copytree(base, target)
    slots = [p for p in model_files(target) if p.suffix == candidate.checkpoint.suffix]
    if len(slots) != 1:
        raise SystemExit(f"{candidate.agent_dir} has {len(slots)} '{candidate.checkpoint.suffix}' "
                         f"model files; cannot tell which one {candidate.checkpoint.name} replaces")
    shutil.copyfile(candidate.checkpoint, slots[0])
    return f"{how}, {slots[0].name} <- {candidate.checkpoint.name}"


# --------------------------------------------------------------------------
# Matches
# --------------------------------------------------------------------------


def lineups(names: Sequence[str], matches: Sequence[str]) -> Dict[str, List[List[str]]]:
    """Tables per match. ``baseline`` and ``mixed`` have one table per candidate."""
    tables: Dict[str, List[List[str]]] = {}
    if "baseline" in matches:
        tables["baseline"] = [[name] + ["rule_based_agent"] * 3 for name in names]
    if "mixed" in matches:
        tables["mixed"] = [[name] + MIXED_FIELD for name in names]
    if "rivals" in matches and len(names) >= 2:
        if len(names) <= 4:
            tables["rivals"] = [list(names) + ["rule_based_agent"] * (4 - len(names))]
        else:
            tables["rivals"] = [list(table) for table in itertools.combinations(names, 4)]
    return tables


def copy_framework(source: Path, target: Path) -> None:
    """The upstream files only: no ``.git``, no agents left over from earlier sims."""
    agent_root = str(source / "agent_code")

    def ignore(directory: str, entries: List[str]) -> List[str]:
        skipped = [e for e in entries if e in (".git", "__pycache__", "replays")]
        if directory == agent_root:
            skipped += [e for e in entries if e not in ts.PROVIDED_AGENTS]
        return skipped

    shutil.copytree(source, target, ignore=ignore)
    shutil.rmtree(target / "logs", ignore_errors=True)
    (target / "logs").mkdir()


def play_round(slot: Path, python: str, match: str, table: Sequence[str], seed: int,
               candidates: Sequence[str]) -> dict:
    """One round in one process; per-agent figures for that round."""
    stats_path = slot / "stats.json"
    stats_path.unlink(missing_ok=True)
    command = [python, "main.py", "play", "--no-gui", "--n-rounds", "1", "--seed", str(seed),
               "--agents", *table, "--save-stats", str(stats_path)]
    record = {"match": match, "table": list(table), "seed": seed, "failed": None}
    try:
        completed = subprocess.run(command, cwd=slot, capture_output=True, text=True, timeout=1800,
                                   env={**ts.tournament_environment(), **SINGLE_THREAD})
        if completed.returncode != 0:
            record["failed"] = f"exit {completed.returncode}: {completed.stderr[-600:]}"
        elif "Traceback" in completed.stderr:
            record["failed"] = f"traceback: {completed.stderr[-600:]}"
    except subprocess.TimeoutExpired:
        record["failed"] = "timeout"

    if record["failed"] is None:
        by_agent = json.loads(stats_path.read_text())["by_agent"]
        record["agents"] = {name: {m: float(values.get(m, 0)) for m in METRICS}
                            for name, values in by_agent.items()}
        record["overruns"] = ts.count_overruns(slot)
    record["log_problems"] = {name: problems for name in candidates if name in table
                              for problems in [ts.agent_errors(slot, name)] if problems}
    ts.clean_agent_logs(slot)
    return record


def play_all(jobs: List[Tuple[str, List[str], int]], slots: List[Path], python: str,
             candidates: Sequence[str]) -> List[dict]:
    free: "queue.Queue[Path]" = queue.Queue()
    for slot in slots:
        free.put(slot)

    def run(job: Tuple[str, List[str], int]) -> dict:
        slot = free.get()
        try:
            return play_round(slot, python, job[0], job[1], job[2], candidates)
        finally:
            free.put(slot)

    records: List[dict] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} games, {time.perf_counter() - started:.0f}s", flush=True)
    return records


# --------------------------------------------------------------------------
# Read-out
# --------------------------------------------------------------------------


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 4000) -> Tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def per_seed_scores(records: Sequence[dict], name: str) -> Dict[str, Dict[int, float]]:
    """match -> seed -> score of ``name`` (mean over tables when it sat at several)."""
    collected: Dict[str, Dict[int, List[float]]] = {}
    for record in records:
        if record["failed"] is None and name in record["agents"]:
            collected.setdefault(record["match"], {}).setdefault(record["seed"], []).append(
                record["agents"][name]["score"])
    return {match: {seed: float(np.mean(scores)) for seed, scores in by_seed.items()}
            for match, by_seed in collected.items()}


def summarise(records: Sequence[dict], names: Sequence[str], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    summary: Dict[str, dict] = {}
    overall: Dict[str, Dict[int, float]] = {}

    for name in names:
        scores = per_seed_scores(records, name)
        mine = [r for r in records if name in r["table"]]
        played = [r for r in mine if r["failed"] is None]
        steps = sum(r["agents"][name]["steps"] for r in played)
        row = {
            "matches": {},
            "games": len(mine),
            "failed_games": sum(1 for r in mine if r["failed"] is not None),
            "log_problems": sorted({p for r in mine for p in r["log_problems"].get(name, [])}),
            "overruns": sum(r["overruns"].get(name, 0) for r in played),
            "think_ms_per_step": 1000.0 * sum(r["agents"][name]["time"] for r in played) / max(1.0, steps),
        }
        for metric in ("coins", "kills", "suicides"):
            row[metric] = float(np.mean([r["agents"][name][metric] for r in played])) if played else float("nan")
        # A round counts as won when nobody at the table scored more.
        row["won"] = float(np.mean([r["agents"][name]["score"] >= max(a["score"] for a in r["agents"].values())
                                    for r in played])) if played else float("nan")
        for match, by_seed in scores.items():
            values = np.array(list(by_seed.values()))
            row["matches"][match] = {"score": float(values.mean()), "ci": bootstrap_ci(values, rng),
                                     "rounds": len(values)}
        # Overall: seeds present in every match, equal weight per match.
        common = set.intersection(*(set(by_seed) for by_seed in scores.values())) if scores else set()
        overall[name] = {s: float(np.mean([scores[m][s] for m in scores])) for s in sorted(common)}
        values = np.array(list(overall[name].values()))
        row["overall"] = {"score": float(values.mean()) if len(values) else float("nan"),
                          "ci": bootstrap_ci(values, rng), "rounds": len(values)}
        row["eligible"] = bool(len(values)) and not row["failed_games"] and not row["log_problems"]
        summary[name] = row

    ranking = sorted(names, key=lambda n: (-summary[n]["eligible"], -np.nan_to_num(summary[n]["overall"]["score"], nan=-1e9)))
    verdict = {"ranking": ranking, "submit": None, "lead": [], "clear": False}
    eligible = [n for n in ranking if summary[n]["eligible"]]
    if eligible:
        best = eligible[0]
        verdict["submit"] = best
        for other in eligible[1:]:
            shared = sorted(set(overall[best]) & set(overall[other]))
            diff = np.array([overall[best][s] - overall[other][s] for s in shared])
            low, high = bootstrap_ci(diff, rng)
            verdict["lead"].append({"over": other, "diff": float(diff.mean()), "ci": (low, high),
                                    "rounds": len(shared)})
        verdict["clear"] = all(lead["ci"][0] > 0 and lead["diff"] >= SCORE_THRESHOLD
                               for lead in verdict["lead"])
    return {"per_candidate": summary, "verdict": verdict}


def format_report(result: dict) -> str:
    summary, verdict = result["per_candidate"], result["verdict"]
    matches = [m for m in MATCHES if any(m in row["matches"] for row in summary.values())]
    header = (f"{'candidate':<44} {'overall':>8} {'95% CI':>16} "
              + " ".join(f"{m:>9}" for m in matches)
              + f" {'won':>5} {'kills':>6} {'suic':>5} {'ms/step':>8} {'overrun':>8}  status")
    lines = [header, "-" * len(header)]
    for name in verdict["ranking"]:
        row = summary[name]
        low, high = row["overall"]["ci"]
        status = "ok" if row["eligible"] else "NOT ELIGIBLE"
        if row["failed_games"]:
            status += f" ({row['failed_games']} crashed)"
        if row["log_problems"]:
            status += " (log: " + "; ".join(row["log_problems"]) + ")"
        if row["eligible"] and row["overruns"]:
            status = "ok, overruns: check latency on a quiet machine"
        lines.append(
            f"{name:<44} {row['overall']['score']:>8.2f} {f'[{low:.2f}, {high:.2f}]':>16} "
            + " ".join(f"{row['matches'][m]['score']:>9.2f}" if m in row["matches"] else f"{'-':>9}"
                       for m in matches)
            + f" {row['won']:>5.2f} {row['kills']:>6.2f} {row['suicides']:>5.2f} "
              f"{row['think_ms_per_step']:>8.2f} {row['overruns']:>8d}  {status}")
    lines.append("")
    if verdict["submit"] is None:
        lines.append("No eligible candidate.")
        return "\n".join(lines)
    lines.append(f"SUBMIT: {verdict['submit']}")
    for lead in verdict["lead"]:
        low, high = lead["ci"]
        lines.append(f"  lead over {lead['over']}: {lead['diff']:+.2f} [{low:+.2f}, {high:+.2f}] "
                     f"(paired, {lead['rounds']} rounds)")
    if verdict["lead"]:
        lines.append("  The lead is clear." if verdict["clear"] else
                     f"  The lead is NOT clear (CI includes 0 or the difference is under "
                     f"{SCORE_THRESHOLD}): keep the incumbent, or play more rounds.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES,
                        help="dir, dir@checkpoint or label=dir@checkpoint")
    parser.add_argument("--matches", nargs="+", default=list(MATCHES), choices=MATCHES)
    parser.add_argument("--rounds", type=int, default=100, help="rounds (= arena seeds) per table")
    parser.add_argument("--seed", type=int, default=20260921, help="seed of the first round")
    parser.add_argument("--jobs", type=int, default=16, help="games played at the same time")
    parser.add_argument("--framework", type=Path, default=ts.DEFAULT_FRAMEWORK,
                        help="pristine upstream checkout (cloned if absent); it is copied, not used")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="where the framework copies go (default: a temporary directory)")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--label", default=None, help="tag for the output file names")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.candidates is DEFAULT_CANDIDATES:
        # Play with the default candidates that exist in this checkout.
        args.candidates = [c for c in DEFAULT_CANDIDATES if (REPO_ROOT / "agent_code" / c).is_dir()]
    candidates = [parse_candidate(spec) for spec in args.candidates]
    names = [c.name for c in candidates]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate candidate names: {names}")

    label = args.label or time.strftime("%Y%m%d-%H%M%S")
    output = OUTPUT_DIR / f"competition__{label}.json"
    if output.exists():
        raise SystemExit(f"{output} exists; results are never overwritten, pick another --label")

    framework = args.framework.resolve()
    ts.ensure_framework(framework)
    if args.workdir:
        args.workdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="competition_", dir=args.workdir))
    try:
        master = workdir / "master"
        copy_framework(framework, master)
        identity = {}
        for candidate in candidates:
            how = stage_candidate(candidate, master, args.python)
            identity[candidate.name] = {"agent_dir": candidate.agent_dir, "staged": how,
                                        "checkpoint": str(candidate.checkpoint) if candidate.checkpoint else None,
                                        "models": fingerprint(master / "agent_code" / candidate.name)}
            print(f"staged {candidate.name}: {how}  {identity[candidate.name]['models']}")

        tables = lineups(names, args.matches)
        jobs = [(match, table, args.seed + i)
                for match, match_tables in tables.items() for table in match_tables
                for i in range(args.rounds)]
        slots = []
        for index in range(max(1, min(args.jobs, len(jobs)))):
            slots.append(workdir / f"slot{index:02d}")
            shutil.copytree(master, slots[-1])
        print(f"{len(jobs)} games on {len(slots)} framework copies in {workdir}")

        started = time.perf_counter()
        records = play_all(jobs, slots, args.python, names)
        elapsed = time.perf_counter() - started
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    result = summarise(records, names)
    report = format_report(result)
    print("\n" + report)
    for record in records:
        if record["failed"]:
            print(f"FAILED {record['match']} seed {record['seed']} {record['table']}: {record['failed']}",
                  file=sys.stderr)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "candidates": identity, "tables": tables, "rounds": args.rounds, "seed": args.seed,
        "jobs": len(slots), "elapsed_seconds": elapsed,
        "framework_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=framework,
                                           capture_output=True, text=True).stdout.strip(),
        **result, "games": records,
    }, indent=1))
    output.with_suffix(".txt").write_text(report + "\n")
    print(f"\nwrote {output.relative_to(REPO_ROOT)} (+ .txt)")
    return 0 if result["verdict"]["submit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

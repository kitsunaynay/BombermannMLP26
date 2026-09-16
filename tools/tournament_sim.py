"""
    python tools/tournament_sim.py --agent attackontensor_ppo
    python tools/tournament_sim.py --agent attackontensor_ql --matches prerun
    python tools/tournament_sim.py --agent attackontensor_ppo --rounds 50 --students survival_linear_ppo_v4 attackontensor_ql
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_URL = "https://github.com/ukoethe/bomberman_rl.git"
DEFAULT_FRAMEWORK = REPO_ROOT / ".tournament"
OUTPUT_DIR = REPO_ROOT / "results" / "tournament_sim"

PROVIDED_AGENTS = {
    "random_agent", "peaceful_agent", "coin_collector_agent", "rule_based_agent",
    "tpl_agent", "user_agent",
}
COPY_IGNORE = shutil.ignore_patterns("__pycache__", "logs", "checkpoints", "*.pyc", "*.log")

OVERRUN_PATTERN = re.compile(r"Agent <(?P<name>[^>]+)> exceeded think time")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agent", required=True, help="agent_code directory to submit")
    parser.add_argument("--students", nargs="*", default=["survival_linear_ppo_v4", "attackontensor_ql"],
                        help="other agent_code directories to copy in as tournament opponents")
    parser.add_argument("--matches", nargs="+", default=["prerun", "baseline", "students"],
                        choices=("prerun", "baseline", "students"))
    parser.add_argument("--rounds", type=int, default=20, help="rounds per match (not prerun)")
    parser.add_argument("--seed", type=int, default=20260921, help="world seed for the matches")
    parser.add_argument("--framework", type=Path, default=DEFAULT_FRAMEWORK,
                        help="where the pristine upstream checkout lives (cloned if absent)")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--label", default=None, help="tag for the output file names")
    return parser


# --------------------------------------------------------------------------
# The pristine framework
# --------------------------------------------------------------------------


def ensure_framework(framework: Path) -> None:
    if not (framework / "main.py").is_file():
        print(f"Cloning {UPSTREAM_URL} -> {framework}")
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", UPSTREAM_URL, str(framework)],
                       check=True)

    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            cwd=framework, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise SystemExit(
            f"{framework} is not pristine; these framework files differ from upstream:\n"
            f"{status}\nRun `git -C {framework} checkout -- .` or delete the directory."
        )


def stage_submission(agent: str, framework: Path, python: str) -> Path:
    """Package the agent exactly as for upload, then unpack it like the graders."""
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / f"{agent}.zip"
        subprocess.run(
            [python, str(REPO_ROOT / "tools" / "package_submission.py"),
             "--agent", agent, "--output", str(archive)],
            check=True, cwd=REPO_ROOT,
        )
        with zipfile.ZipFile(archive) as bundle:
            prefix = locate_agent_in_zip(bundle.namelist())
            target = framework / "agent_code" / agent
            if target.exists():
                shutil.rmtree(target)
            for member in bundle.namelist():
                if not member.startswith(prefix) or member.endswith("/"):
                    continue
                destination = target / member[len(prefix):]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(bundle.read(member))
    return target


def locate_agent_in_zip(names: Sequence[str]) -> str:
    """Directory prefix of the first ``callbacks.py`` in the archive (brief, section 8)."""
    candidates = sorted(name for name in names if name.endswith("callbacks.py"))
    if not candidates:
        raise SystemExit("the archive contains no callbacks.py")
    return candidates[0][: -len("callbacks.py")]


def stage_student(code_name: str, framework: Path) -> None:
    if code_name in PROVIDED_AGENTS:
        return
    source = REPO_ROOT / "agent_code" / code_name
    if not source.is_dir():
        raise SystemExit(f"no such agent directory: {source}")
    target = framework / "agent_code" / code_name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=COPY_IGNORE, symlinks=False)


# --------------------------------------------------------------------------
# Matches
# --------------------------------------------------------------------------


def tournament_environment() -> Dict[str, str]:
    """The graders set nothing; make sure no ``AOT_*`` variable leaks in."""
    return {key: value for key, value in os.environ.items() if not key.startswith("AOT_")}


def play(framework: Path, python: str, agents: Sequence[str], rounds: int, seed: int,
         stats_path: Path) -> subprocess.CompletedProcess:
    game_log = framework / "logs" / "game.log"
    if game_log.exists():
        game_log.unlink()
    command = [
        python, "main.py", "play", "--no-gui",
        "--n-rounds", str(rounds), "--seed", str(seed),
        "--agents", *agents,
        "--save-stats", str(stats_path),
    ]
    return subprocess.run(command, cwd=framework, env=tournament_environment(),
                          capture_output=True, text=True, timeout=3600)


def count_overruns(framework: Path) -> Dict[str, int]:
    game_log = framework / "logs" / "game.log"
    counts: Dict[str, int] = {}
    if not game_log.exists():
        return counts
    for line in game_log.read_text(errors="replace").splitlines():
        match = OVERRUN_PATTERN.search(line)
        if match:
            counts[match.group("name")] = counts.get(match.group("name"), 0) + 1
    return counts


def agent_errors(framework: Path, agent: str) -> List[str]:
    """Tracebacks in the agent's own log: what the graders would send back."""
    log_dir = framework / "agent_code" / agent / "logs"
    found: List[str] = []
    for log in log_dir.glob("*.log") if log_dir.is_dir() else []:
        text = log.read_text(errors="replace")
        for marker in ("Traceback", "ERROR", "falling back"):
            if marker in text:
                found.append(f"{log.name}: contains '{marker}'")
    return found


def summarise(payload: dict, overruns: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """Per-agent, per-round figures from ``main.py --save-stats`` output."""
    rounds = max(1, len(payload.get("by_round", {})))
    table: Dict[str, Dict[str, float]] = {}
    for name, values in payload.get("by_agent", {}).items():
        steps = max(1, int(values.get("steps", 0)))
        table[name] = {
            "rounds": rounds,
            "score": float(values.get("score", 0)) / rounds,
            "coins": float(values.get("coins", 0)) / rounds,
            "kills": float(values.get("kills", 0)) / rounds,
            "suicides": float(values.get("suicides", 0)) / rounds,
            "invalid": float(values.get("invalid", 0)) / rounds,
            "steps": steps / rounds,
            "think_ms_per_step": 1000.0 * float(values.get("time", 0.0)) / steps,
            "overruns": overruns.get(name, 0),
        }
    return table


def format_table(table: Dict[str, Dict[str, float]]) -> str:
    header = (f"{'agent':<28} {'score':>7} {'coins':>7} {'kills':>7} {'suic':>7} "
              f"{'steps':>7} {'ms/step':>8} {'overrun':>8}")
    lines = [header, "-" * len(header)]
    for name, row in sorted(table.items(), key=lambda item: -item[1]["score"]):
        lines.append(
            f"{name:<28} {row['score']:>7.2f} {row['coins']:>7.2f} {row['kills']:>7.2f} "
            f"{row['suicides']:>7.2f} {row['steps']:>7.1f} {row['think_ms_per_step']:>8.2f} "
            f"{int(row['overruns']):>8d}"
        )
    return "\n".join(lines)


def clean_agent_logs(framework: Path) -> None:
    """The stock settings log every agent at DEBUG; do not let that pile up."""
    for log_dir in (framework / "agent_code").glob("*/logs"):
        shutil.rmtree(log_dir, ignore_errors=True)


def run_match(name: str, agents: Sequence[str], rounds: int, args, framework: Path,
              stamp: str) -> Optional[dict]:
    stats_path = framework / f"stats_{name}.json"
    print(f"\n=== {name}: {' vs '.join(agents)}  ({rounds} round(s), seed {args.seed}) ===")
    started = time.perf_counter()
    completed = play(framework, args.python, agents, rounds, args.seed, stats_path)
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        print(completed.stdout[-2000:])
        print(completed.stderr[-3000:], file=sys.stderr)
        print(f"FAIL: main.py exited {completed.returncode}")
        return None
    if "Traceback" in completed.stderr:
        print(completed.stderr[-3000:], file=sys.stderr)
        print("FAIL: a traceback reached the console")
        return None

    payload = json.loads(stats_path.read_text())
    overruns = count_overruns(framework)
    table = summarise(payload, overruns)
    errors = agent_errors(framework, args.agent)

    print(format_table(table))
    print(f"wall clock {elapsed:.1f}s")
    if errors:
        print("Agent log problems:\n  " + "\n  ".join(errors))

    result = {
        "match": name,
        "agents": list(agents),
        "rounds": rounds,
        "seed": args.seed,
        "framework_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=framework,
            capture_output=True, text=True).stdout.strip(),
        "elapsed_seconds": elapsed,
        "per_agent": table,
        "agent_log_problems": errors,
        "raw": payload,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label = args.label or stamp
    output = OUTPUT_DIR / f"{args.agent}__{name}__{label}.json"
    output.write_text(json.dumps(result, indent=2))
    print(f"wrote {output.relative_to(REPO_ROOT)}")
    stats_path.unlink(missing_ok=True)
    clean_agent_logs(framework)
    return result


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    framework = args.framework.resolve()
    ensure_framework(framework)

    staged = stage_submission(args.agent, framework, args.python)
    print(f"staged {args.agent} from its submission zip -> {staged}")
    for student in args.students:
        stage_student(student, framework)
    clean_agent_logs(framework)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    failures = 0

    if "prerun" in args.matches:
        result = run_match("prerun", [args.agent] + ["random_agent"] * 3, 1, args, framework, stamp)
        if result is None or result["agent_log_problems"]:
            failures += 1
            print("PRE-RUN CHECK FAILED -- fix this before uploading.")
        else:
            print("pre-run check passed: no errors, no tracebacks.")

    if "baseline" in args.matches:
        if run_match("baseline", [args.agent] + ["rule_based_agent"] * 3, args.rounds,
                     args, framework, stamp) is None:
            failures += 1

    if "students" in args.matches:
        seats = [args.agent, *args.students][:4]
        while len(seats) < 4:
            seats.append("rule_based_agent")
        if run_match("students", seats, args.rounds, args, framework, stamp) is None:
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

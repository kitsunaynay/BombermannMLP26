"""Package one agent directory as the tournament submission zip.
    python tools/package_submission.py --agent attackontensor_ql
    python tools/package_submission.py --agent attackontensor_ppo --output submit.zip
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import zipfile
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: Never shipped. `checkpoints` holds the training snapshots: they are training
#: output, not inference input, and including them made the PPO archive 2.1 GB.
EXCLUDED_DIRS = {
    "__pycache__",
    "logs",
    "checkpoints",
    ".ipynb_checkpoints",
    ".pytest_cache",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp"}

#: An archive larger than this is a packaging mistake, not a big model. The
#: largest thing we legitimately ship is a single `policy.pt` at ~19 MB.
MAX_ARCHIVE_MB = 50.0

#: Modules the framework itself provides at the repository root, so an agent may
#: import them even though they are not inside its directory.
FRAMEWORK_MODULES = {"settings", "events", "items", "environment", "agents", "fallbacks"}

#: Present at the root during development but absent in the tournament.
REPO_ONLY_PACKAGES = {"blib", "shared", "tools", "tests"}

#: Trained parameters the agent needs to actually play.
MODEL_SUFFIXES = {".pkl", ".pt", ".npz", ".npy", ".json"}


def iter_files(agent_dir: Path) -> List[Path]:
    files = []
    for path in sorted(agent_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(agent_dir).parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return files


def check_imports(files: List[Path]) -> List[str]:
    """Flag imports that will not resolve once the directory stands alone."""
    problems = []
    for path in files:
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as error:
            problems.append(f"{path.name}: syntax error at line {error.lineno}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    continue  # relative import, stays inside the package
                root = (node.module or "").split(".")[0]
            elif isinstance(node, ast.Import):
                root = node.names[0].name.split(".")[0]
            else:
                continue

            if root in REPO_ONLY_PACKAGES:
                problems.append(
                    f"{path.name}: imports '{root}', which does not exist in the tournament tree"
                )
    return problems


def check_absolute_paths(files: List[Path]) -> List[str]:
    """Flag string literals that look like machine-specific absolute paths.

    The brief calls this out explicitly: *"A common error is the use of absolute
    paths to reference files in other directories."*
    """
    pattern = re.compile(r"""['"](/(?:Users|home|mnt|tmp|var)/[^'"]*)['"]""")
    problems = []
    for path in files:
        if path.suffix != ".py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            match = pattern.search(line)
            if match:
                problems.append(f"{path.name}:{number}: absolute path {match.group(1)!r}")
    return problems


def check_archive_size(files: List[Path]) -> List[str]:
    """Flag an archive that is far larger than a trained agent needs to be.

    The failure this catches: a stray directory of training snapshots sweeping
    into the zip. Measured before the `checkpoints` exclusion existed, the PPO
    archive was 2149 MB, of which 2130 MB was 111 snapshots.
    """
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    if total_mb <= MAX_ARCHIVE_MB:
        return []
    largest = sorted(files, key=lambda f: f.stat().st_size, reverse=True)[:3]
    listing = ", ".join(f"{f.name} ({f.stat().st_size / 1e6:.0f} MB)" for f in largest)
    return [
        f"uncompressed size {total_mb:.0f} MB exceeds {MAX_ARCHIVE_MB:.0f} MB; "
        f"largest entries: {listing}"
    ]


def check_model_present(agent_dir: Path, files: List[Path]) -> List[str]:
    models = [f for f in files if f.suffix in MODEL_SUFFIXES and f.name != "config.json"]
    if not models:
        return [
            "no trained parameters found (.pkl/.pt/.npz) -- the agent would play "
            "from an untrained model"
        ]
    return []


def check_required_callbacks(agent_dir: Path) -> List[str]:
    problems = []
    callbacks = agent_dir / "callbacks.py"
    if not callbacks.is_file():
        return ["callbacks.py is missing; the graders locate the agent by that file"]

    source = callbacks.read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    for name, expected in (("setup", 1), ("act", 2)):
        node = functions.get(name)
        if node is None:
            problems.append(f"callbacks.py is missing '{name}'")
        elif len(node.args.args) != expected:
            # AgentRunner checks arity at load time and refuses to start.
            problems.append(
                f"callbacks.py '{name}' takes {len(node.args.args)} args, needs {expected}"
            )
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agent", required=True, help="agent_code directory name")
    parser.add_argument("--output", default=None, help="zip path")
    parser.add_argument("--force", action="store_true", help="package despite warnings")
    args = parser.parse_args(argv)

    agent_dir = REPO_ROOT / "agent_code" / args.agent
    if not agent_dir.is_dir():
        print(f"No such agent directory: {agent_dir}", file=sys.stderr)
        return 2

    files = iter_files(agent_dir)
    if not files:
        print(f"{agent_dir} contains nothing to package", file=sys.stderr)
        return 2

    problems = (
        check_required_callbacks(agent_dir)
        + check_imports(files)
        + check_absolute_paths(files)
        + check_archive_size(files)
    )
    warnings = check_model_present(agent_dir, files)

    print(f"Packaging {args.agent} ({len(files)} files)")
    total = sum(f.stat().st_size for f in files)
    print(f"  uncompressed size: {total / 1e6:.1f} MB")

    for problem in problems:
        print(f"  ERROR   {problem}")
    for warning in warnings:
        print(f"  WARNING {warning}")

    if problems and not args.force:
        print("\nRefusing to package. Fix the errors above, or pass --force.", file=sys.stderr)
        return 1
    if warnings and not args.force:
        print("\nRefusing to package an agent with no trained parameters. "
              "Train it first, or pass --force.", file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else REPO_ROOT / "final-project-agent-code.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            # Store paths as "<agent>/..." so the graders' "first directory
            # containing callbacks.py" is exactly this agent.
            archive.write(path, Path(args.agent) / path.relative_to(agent_dir))

    print(f"\nWrote {output.name} ({output.stat().st_size / 1e6:.1f} MB)")
    print("Upload this to MaMPF. Reminder: also test it with `docker build .`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

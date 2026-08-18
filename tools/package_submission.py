#!/usr/bin/env python3
"""Package one agent directory as the tournament submission zip.

The brief describes what the graders do with the upload:

1. unzip it;
2. install anything in ``requirements.txt``;
3. **find the first directory containing a `callbacks.py`**;
4. copy that directory into their own `agent_code`;
5. run one game with ``self.train = False`` against three ``random_agent``s.

So the archive has to contain exactly one agent directory, that directory has to
be self-contained, and it must carry its trained parameters. This script builds
that archive, leaves out caches and logs, and verifies the result before writing
it -- catching the failure modes that only show up on the graders' machine:
a missing model file, a stray import from the repository root, or an absolute
path baked into the code.

Usage::

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

#: Never shipped.
EXCLUDED_DIRS = {"__pycache__", "logs", ".ipynb_checkpoints", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp"}

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

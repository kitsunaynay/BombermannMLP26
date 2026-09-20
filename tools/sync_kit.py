"""Vendor ``shared/kit`` into each agent directory.
    python tools/sync_kit.py            # write the copies
    python tools/sync_kit.py --check    # verify, exit 1 on drift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "shared" / "kit"

#: Agent directories that receive a vendored copy.
AGENT_DIRS: Tuple[str, ...] = (
    "attackontensor_ql",
    "attackontensor_ppo",
)

BANNER_MARK = "# GENERATED FILE -- DO NOT EDIT."


def banner(source_name: str) -> str:
    return (
        "# " + "-" * 74 + "\n"
        f"{BANNER_MARK}\n"
        f"# Vendored from shared/kit/{source_name} by tools/sync_kit.py.\n"
        "# Edit the original, then re-run:  python tools/sync_kit.py\n"
        "# " + "-" * 74 + "\n"
    )


def strip_banner(text: str) -> str:
    """Remove a generated banner, so copies compare equal to their source."""
    if BANNER_MARK not in text:
        return text
    lines = text.splitlines(keepends=True)
    # The banner is a contiguous run of comment lines at the very top.
    for index, line in enumerate(lines):
        if not line.startswith("#"):
            return "".join(lines[index:])
    return ""


def source_files() -> List[Path]:
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"Source kit not found at {SOURCE_DIR}")
    return sorted(SOURCE_DIR.glob("*.py"))


def target_dir(agent: str) -> Path:
    return REPO_ROOT / "agent_code" / agent / "kit"


def sync(check_only: bool = False) -> int:
    problems: List[str] = []
    written = 0

    for agent in AGENT_DIRS:
        destination = target_dir(agent)
        if not check_only:
            destination.mkdir(parents=True, exist_ok=True)

        for source in source_files():
            expected = banner(source.name) + source.read_text()
            target = destination / source.name

            if check_only:
                if not target.exists():
                    problems.append(f"missing: {target.relative_to(REPO_ROOT)}")
                elif strip_banner(target.read_text()) != source.read_text():
                    problems.append(f"drifted: {target.relative_to(REPO_ROOT)}")
                continue

            if not target.exists() or target.read_text() != expected:
                target.write_text(expected)
                written += 1

        # Remove vendored files whose source was deleted.
        if destination.is_dir():
            valid = {p.name for p in source_files()}
            for stale in destination.glob("*.py"):
                if stale.name not in valid:
                    if check_only:
                        problems.append(f"stale: {stale.relative_to(REPO_ROOT)}")
                    else:
                        stale.unlink()

    if check_only:
        if problems:
            print("Vendored kit is out of sync:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            print("\nRun: python tools/sync_kit.py", file=sys.stderr)
            return 1
        print(f"Vendored kit is in sync across {len(AGENT_DIRS)} agent(s).")
        return 0

    print(f"Synced shared/kit -> {len(AGENT_DIRS)} agent(s); {written} file(s) updated.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the copies match without writing; exit 1 on drift",
    )
    args = parser.parse_args()
    return sync(check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())

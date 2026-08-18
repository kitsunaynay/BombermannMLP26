"""Guard against vendored-kit drift.

``shared/kit`` is copied into each agent directory because the tournament only
receives that one directory. Duplication without a check is how the copies quietly
diverge, so the check runs as part of the test suite.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import sync_kit  # noqa: E402


def test_vendored_kit_is_in_sync():
    assert sync_kit.sync(check_only=True) == 0, (
        "Vendored kit copies differ from shared/kit. Run: python tools/sync_kit.py"
    )


def test_every_agent_gets_every_module():
    for agent in sync_kit.AGENT_DIRS:
        destination = sync_kit.target_dir(agent)
        vendored = {p.name for p in destination.glob("*.py")}
        expected = {p.name for p in sync_kit.source_files()}
        assert vendored == expected, f"{agent} kit contents differ from shared/kit"

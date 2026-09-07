"""Tests for the submission packager and the report asset generator.

The packager's checks encode the failure modes that only surface on the graders'
machine, where the agent directory stands alone: a repo-root import that no
longer resolves, an absolute path baked into the source, a missing model file, or
a callback with the wrong arity. Each check is tested against a deliberately
broken agent so a regression in the check itself does not pass silently.
"""

import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import package_submission as pkg  # noqa: E402


def make_agent(tmp_path, callbacks_source: str, with_model: bool = True) -> Path:
    agent_dir = tmp_path / "agent_code" / "demo_agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "callbacks.py").write_text(callbacks_source)
    if with_model:
        (agent_dir / "model.pkl").write_bytes(b"weights")
    return agent_dir


GOOD_CALLBACKS = """
import numpy as np
import settings as s
from .kit import geometry


def setup(self):
    pass


def act(self, game_state):
    return "WAIT"
"""


def test_valid_agent_passes_every_check(tmp_path):
    agent_dir = make_agent(tmp_path, GOOD_CALLBACKS)
    files = pkg.iter_files(agent_dir)

    assert pkg.check_required_callbacks(agent_dir) == []
    assert pkg.check_imports(files) == []
    assert pkg.check_absolute_paths(files) == []
    assert pkg.check_model_present(agent_dir, files) == []


def test_repo_only_import_is_flagged(tmp_path):
    """`import blib` works in development and vanishes in the tournament."""
    agent_dir = make_agent(tmp_path, "from blib.fast_env import FastWorld\n" + GOOD_CALLBACKS)

    problems = pkg.check_imports(pkg.iter_files(agent_dir))

    assert any("blib" in problem for problem in problems)


def test_relative_imports_are_not_flagged(tmp_path):
    agent_dir = make_agent(tmp_path, "from .kit import geometry\nfrom . import config\n" + GOOD_CALLBACKS)
    assert pkg.check_imports(pkg.iter_files(agent_dir)) == []


def test_framework_imports_are_allowed(tmp_path):
    """settings and events live at the repository root in the tournament too."""
    agent_dir = make_agent(tmp_path, "import settings as s\nimport events as e\n" + GOOD_CALLBACKS)
    assert pkg.check_imports(pkg.iter_files(agent_dir)) == []


def test_absolute_path_is_flagged(tmp_path):
    """The brief calls absolute paths out as a common submission failure."""
    source = GOOD_CALLBACKS + '\nMODEL = "/Users/someone/models/policy.pt"\n'
    agent_dir = make_agent(tmp_path, source)

    problems = pkg.check_absolute_paths(pkg.iter_files(agent_dir))

    assert len(problems) == 1 and "/Users/someone" in problems[0]


def test_absolute_path_inside_a_comment_is_ignored(tmp_path):
    source = GOOD_CALLBACKS + '\n# see /Users/someone/notes.txt for details\n'
    agent_dir = make_agent(tmp_path, source)
    assert pkg.check_absolute_paths(pkg.iter_files(agent_dir)) == []


def test_missing_model_is_reported(tmp_path):
    agent_dir = make_agent(tmp_path, GOOD_CALLBACKS, with_model=False)
    problems = pkg.check_model_present(agent_dir, pkg.iter_files(agent_dir))
    assert problems and "trained parameters" in problems[0]


def test_config_json_alone_does_not_count_as_a_model(tmp_path):
    agent_dir = make_agent(tmp_path, GOOD_CALLBACKS, with_model=False)
    (agent_dir / "config.json").write_text("{}")

    assert pkg.check_model_present(agent_dir, pkg.iter_files(agent_dir))


def test_missing_callback_is_reported(tmp_path):
    agent_dir = make_agent(tmp_path, "def setup(self):\n    pass\n")
    problems = pkg.check_required_callbacks(agent_dir)
    assert any("act" in problem for problem in problems)


def test_wrong_callback_arity_is_reported(tmp_path):
    """AgentRunner checks arity at load time and refuses to start (agents.py:214)."""
    agent_dir = make_agent(tmp_path, "def setup(self):\n    pass\n\ndef act(self):\n    return 'WAIT'\n")

    problems = pkg.check_required_callbacks(agent_dir)

    assert any("act" in problem and "needs 2" in problem for problem in problems)


def test_iter_files_excludes_caches_and_logs(tmp_path):
    agent_dir = make_agent(tmp_path, GOOD_CALLBACKS)
    (agent_dir / "__pycache__").mkdir()
    (agent_dir / "__pycache__" / "callbacks.cpython-312.pyc").write_bytes(b"x")
    (agent_dir / "logs").mkdir()
    (agent_dir / "logs" / "agent.log").write_text("noise")

    names = {path.name for path in pkg.iter_files(agent_dir)}

    assert names == {"callbacks.py", "model.pkl"}


def test_iter_files_excludes_training_checkpoints(tmp_path):
    """Snapshots are training output; shipping them made the PPO zip 2.1 GB."""
    agent_dir = make_agent(tmp_path, GOOD_CALLBACKS)
    (agent_dir / "checkpoints").mkdir()
    (agent_dir / "checkpoints" / "run-a").mkdir()
    (agent_dir / "checkpoints" / "run-a" / "policy_it000040.pt").write_bytes(b"x")
    (agent_dir / "checkpoints" / "q_table_r000250.pkl").write_bytes(b"x")

    names = {path.name for path in pkg.iter_files(agent_dir)}

    assert names == {"callbacks.py", "model.pkl"}


def test_oversized_archive_is_flagged(tmp_path):
    agent_dir = make_agent(tmp_path, GOOD_CALLBACKS)
    oversized = int(pkg.MAX_ARCHIVE_MB * 1e6) + 1
    (agent_dir / "policy.pt").write_bytes(b"\0" * oversized)

    problems = pkg.check_archive_size(pkg.iter_files(agent_dir))

    assert len(problems) == 1
    assert "exceeds" in problems[0]
    assert "policy.pt" in problems[0]


def test_normal_archive_is_not_flagged(tmp_path):
    agent_dir = make_agent(tmp_path, GOOD_CALLBACKS)

    assert pkg.check_archive_size(pkg.iter_files(agent_dir)) == []


def test_real_agents_pass_their_own_checks():
    """The shipped agents must satisfy the packager they are packaged by."""
    for name in ("attackontensor_ql", "attackontensor_ppo"):
        agent_dir = REPO_ROOT / "agent_code" / name
        files = pkg.iter_files(agent_dir)

        assert pkg.check_required_callbacks(agent_dir) == [], name
        assert pkg.check_imports(files) == [], name
        assert pkg.check_absolute_paths(files) == [], name
        assert pkg.check_archive_size(files) == [], name


def test_zip_is_rooted_at_the_agent_directory(tmp_path, monkeypatch):
    """The graders take the first directory containing callbacks.py."""
    agent_dir = make_agent(tmp_path, GOOD_CALLBACKS)
    monkeypatch.setattr(pkg, "REPO_ROOT", tmp_path)

    output = tmp_path / "submission.zip"
    assert pkg.main(["--agent", "demo_agent", "--output", str(output)]) == 0

    names = zipfile.ZipFile(output).namelist()
    assert all(name.startswith("demo_agent/") for name in names)
    assert "demo_agent/callbacks.py" in names


def test_packaging_refuses_a_broken_agent(tmp_path, monkeypatch):
    make_agent(tmp_path, "from tools.sync_kit import sync\n" + GOOD_CALLBACKS)
    monkeypatch.setattr(pkg, "REPO_ROOT", tmp_path)

    code = pkg.main(["--agent", "demo_agent", "--output", str(tmp_path / "out.zip")])

    assert code == 1
    assert not (tmp_path / "out.zip").exists()


def test_force_overrides_the_refusal(tmp_path, monkeypatch):
    make_agent(tmp_path, GOOD_CALLBACKS, with_model=False)
    monkeypatch.setattr(pkg, "REPO_ROOT", tmp_path)

    output = tmp_path / "out.zip"
    assert pkg.main(["--agent", "demo_agent", "--output", str(output), "--force"]) == 0
    assert output.exists()

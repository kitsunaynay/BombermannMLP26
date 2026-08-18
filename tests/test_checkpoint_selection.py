"""Best-checkpoint selection.

A constant learning rate makes tabular Q-values track a moving target rather
than converge, so the table a run writes last is not reliably the best table the
run produced. Measured on Task 2: the final table scored 16.3 coins where a
table from 2 000 rounds earlier scored 24.9, on the same seeds. Trusting the
last write was the defect; these tests pin the fix.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import train_ql  # noqa: E402

from agent_code.attackontensor_ql.qtable import QTable  # noqa: E402
from blib.curriculum import get_stage  # noqa: E402


@pytest.fixture
def fake_agent_dir(tmp_path, monkeypatch):
    """An agent directory with snapshots, isolated from the real one."""
    agent_dir = tmp_path / "agent"
    (agent_dir / "checkpoints").mkdir(parents=True)
    monkeypatch.setattr(train_ql, "AGENT_DIR", agent_dir)
    return agent_dir


def write_table(path: Path, marker: int) -> None:
    table = QTable(double=False, seed=0)
    table.metadata.update(variant="full", use_symmetry=True, marker=marker)
    table.learn((marker,), 0, float(marker), None, 0.0, 1.0)
    table.save(path)


def args_for(select_seeds=2):
    import argparse

    return argparse.Namespace(select_seeds=select_seeds)


def test_selection_promotes_the_best_scoring_snapshot(fake_agent_dir, tmp_path, monkeypatch):
    """The winner on held-out seeds becomes the live model."""
    write_table(fake_agent_dir / "checkpoints" / "q_table_r000250.pkl", 1)
    write_table(fake_agent_dir / "checkpoints" / "q_table_r000500.pkl", 2)
    write_table(fake_agent_dir / "q_table.pkl", 3)

    scores = {
        "q_table_r000250.pkl": 5.0,
        "q_table_r000500.pkl": 42.0,  # the winner
        "q_table.pkl": 9.0,
    }
    seen = []

    def fake_benchmark(config, progress=False):
        import os

        name = Path(os.environ["AOT_QL_MODEL_FILE"]).name
        seen.append((name, config.base_seed))
        return {
            "per_agent": {
                "attackontensor_ql_0": {
                    "score_mean": scores[name],
                    "survival_rate": 1.0,
                    "suicide_rate": 0.0,
                }
            }
        }

    monkeypatch.setattr(train_ql, "benchmark_fast", fake_benchmark)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_ql.select_best_checkpoint(args_for(), get_stage(2), run_dir)

    promoted = QTable.load(fake_agent_dir / "q_table.pkl")
    assert promoted.metadata["marker"] == 2, "the highest-scoring snapshot must win"

    # Every candidate was evaluated, and on the selection seed set.
    assert {name for name, _ in seen} == set(scores)
    assert all(seed == train_ql.SELECTION_BASE_SEED for _, seed in seen)


def test_selection_uses_seeds_held_out_from_the_report(fake_agent_dir, tmp_path, monkeypatch):
    """Choosing and reporting on the same arenas would inflate the result."""
    from blib.benchmark import MatchConfig

    assert train_ql.SELECTION_BASE_SEED != MatchConfig(agents=["x"]).base_seed


def test_selection_breaks_ties_on_survival(fake_agent_dir, tmp_path, monkeypatch):
    """Equal score, fewer deaths -- the better tournament agent."""
    write_table(fake_agent_dir / "checkpoints" / "q_table_r000250.pkl", 1)
    write_table(fake_agent_dir / "q_table.pkl", 2)

    values = {
        "q_table_r000250.pkl": {"score_mean": 10.0, "survival_rate": 0.9, "suicide_rate": 0.1},
        "q_table.pkl": {"score_mean": 10.0, "survival_rate": 0.4, "suicide_rate": 0.6},
    }

    def fake_benchmark(config, progress=False):
        import os

        name = Path(os.environ["AOT_QL_MODEL_FILE"]).name
        return {"per_agent": {"attackontensor_ql_0": values[name]}}

    monkeypatch.setattr(train_ql, "benchmark_fast", fake_benchmark)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_ql.select_best_checkpoint(args_for(), get_stage(2), run_dir)

    assert QTable.load(fake_agent_dir / "q_table.pkl").metadata["marker"] == 1


def test_selection_writes_an_audit_trail(fake_agent_dir, tmp_path, monkeypatch):
    write_table(fake_agent_dir / "checkpoints" / "q_table_r000250.pkl", 1)
    write_table(fake_agent_dir / "q_table.pkl", 2)

    monkeypatch.setattr(
        train_ql,
        "benchmark_fast",
        lambda config, progress=False: {
            "per_agent": {"attackontensor_ql_0": {"score_mean": 1.0, "survival_rate": 1.0}}
        },
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_ql.select_best_checkpoint(args_for(), get_stage(2), run_dir)

    record = json.loads((run_dir / "checkpoint_selection.json").read_text())
    assert len(record) == 2
    assert {"checkpoint", "score", "survival"} <= set(record[0])


def test_selection_is_skipped_with_a_single_checkpoint(fake_agent_dir, tmp_path, monkeypatch):
    write_table(fake_agent_dir / "q_table.pkl", 1)

    called = []
    monkeypatch.setattr(
        train_ql, "benchmark_fast", lambda *a, **k: called.append(1) or {"per_agent": {}}
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_ql.select_best_checkpoint(args_for(), get_stage(2), run_dir)

    assert not called, "nothing to choose between; do not waste a benchmark run"


def test_selection_does_not_leak_the_model_env_var(fake_agent_dir, tmp_path, monkeypatch):
    """AOT_QL_MODEL_FILE is set per candidate and must not survive the call.

    Leaving it set would silently redirect the *reporting* benchmark that runs
    immediately afterwards to whichever checkpoint happened to be evaluated last.
    """
    import os

    write_table(fake_agent_dir / "checkpoints" / "q_table_r000250.pkl", 1)
    write_table(fake_agent_dir / "q_table.pkl", 2)

    monkeypatch.setattr(
        train_ql,
        "benchmark_fast",
        lambda config, progress=False: {
            "per_agent": {"attackontensor_ql_0": {"score_mean": 1.0, "survival_rate": 1.0}}
        },
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_ql.select_best_checkpoint(args_for(), get_stage(2), run_dir)

    assert "AOT_QL_MODEL_FILE" not in os.environ

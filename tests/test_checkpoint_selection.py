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
    """An agent directory isolated from the real one."""
    agent_dir = tmp_path / "agent"
    (agent_dir / "checkpoints").mkdir(parents=True)
    monkeypatch.setattr(train_ql, "AGENT_DIR", agent_dir)
    return agent_dir


@pytest.fixture
def work_dir(fake_agent_dir):
    """Per-run artifact directory: snapshots and the run's live table."""
    directory = fake_agent_dir / "checkpoints" / "testrun"
    directory.mkdir(parents=True)
    return directory


def write_table(path: Path, marker: int) -> None:
    table = QTable(double=False, seed=0)
    table.metadata.update(variant="full", use_symmetry=True, marker=marker)
    table.learn((marker,), 0, float(marker), None, 0.0, 1.0)
    table.save(path)


def args_for(select_seeds=2):
    import argparse

    return argparse.Namespace(select_seeds=select_seeds)


def test_selection_returns_the_best_scoring_snapshot(fake_agent_dir, work_dir, tmp_path, monkeypatch):
    """The winner on held-out seeds is the one returned."""
    write_table(work_dir / "q_table_r000250.pkl", 1)
    write_table(work_dir / "q_table_r000500.pkl", 2)
    write_table(work_dir / "q_table.pkl", 3)

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
    best = train_ql.select_best_checkpoint(args_for(), get_stage(2), run_dir, work_dir)

    assert QTable.load(best).metadata["marker"] == 2, "the highest-scoring snapshot must win"
    assert not (fake_agent_dir / "q_table.pkl").exists(), (
        "selection must not write the shipped table; promotion is a separate step"
    )

    # Every candidate was evaluated, and on the selection seed set.
    assert {name for name, _ in seen} == set(scores)
    assert all(seed == train_ql.SELECTION_BASE_SEED for _, seed in seen)


def test_selection_uses_seeds_held_out_from_the_report(fake_agent_dir, tmp_path, monkeypatch):
    """Choosing and reporting on the same arenas would inflate the result."""
    from blib.benchmark import MatchConfig

    assert train_ql.SELECTION_BASE_SEED != MatchConfig(agents=["x"]).base_seed


def test_selection_breaks_ties_on_survival(fake_agent_dir, work_dir, tmp_path, monkeypatch):
    """Equal score, fewer deaths -- the better tournament agent."""
    write_table(work_dir / "q_table_r000250.pkl", 1)
    write_table(work_dir / "q_table.pkl", 2)

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
    best = train_ql.select_best_checkpoint(args_for(), get_stage(2), run_dir, work_dir)

    assert QTable.load(best).metadata["marker"] == 1


def test_selection_writes_an_audit_trail(fake_agent_dir, work_dir, tmp_path, monkeypatch):
    write_table(work_dir / "q_table_r000250.pkl", 1)
    write_table(work_dir / "q_table.pkl", 2)

    monkeypatch.setattr(
        train_ql,
        "benchmark_fast",
        lambda config, progress=False: {
            "per_agent": {"attackontensor_ql_0": {"score_mean": 1.0, "survival_rate": 1.0}}
        },
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    best = train_ql.select_best_checkpoint(args_for(), get_stage(2), run_dir, work_dir)

    record = json.loads((run_dir / "checkpoint_selection.json").read_text())
    assert len(record) == 2
    assert {"checkpoint", "score", "survival"} <= set(record[0])


def test_selection_is_skipped_with_a_single_checkpoint(fake_agent_dir, work_dir, tmp_path, monkeypatch):
    write_table(work_dir / "q_table.pkl", 1)

    called = []
    monkeypatch.setattr(
        train_ql, "benchmark_fast", lambda *a, **k: called.append(1) or {"per_agent": {}}
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    best = train_ql.select_best_checkpoint(args_for(), get_stage(2), run_dir, work_dir)

    assert not called, "nothing to choose between; do not waste a benchmark run"
    assert best == work_dir / "q_table.pkl"


def test_selection_does_not_leak_the_model_env_var(fake_agent_dir, work_dir, tmp_path, monkeypatch):
    """AOT_QL_MODEL_FILE is set per candidate and must not survive the call.

    Leaving it set would silently redirect the *reporting* benchmark that runs
    immediately afterwards to whichever checkpoint happened to be evaluated last.
    """
    import os

    write_table(work_dir / "q_table_r000250.pkl", 1)
    write_table(work_dir / "q_table.pkl", 2)

    monkeypatch.setattr(
        train_ql,
        "benchmark_fast",
        lambda config, progress=False: {
            "per_agent": {"attackontensor_ql_0": {"score_mean": 1.0, "survival_rate": 1.0}}
        },
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    best = train_ql.select_best_checkpoint(args_for(), get_stage(2), run_dir, work_dir)

    assert "AOT_QL_MODEL_FILE" not in os.environ


def test_selection_prefers_the_more_trained_table_on_a_tie(
    fake_agent_dir, work_dir, tmp_path, monkeypatch
):
    """Equal score and equal survival: take the table with more training.

    Ties are common once a stage is solved -- on Task 1 nine checkpoints scored
    exactly 50.00/50 with 100% survival. A stable sort keeps glob order, so the
    winner was the *earliest* snapshot: a 92-state table with 16k visits
    promoted over an equally perfect 130-state one with 368k. Harmless for the
    stage that tied, but it became the seed for the next stage.
    """
    write_table(work_dir / "q_table_r000250.pkl", 1)
    write_table(work_dir / "q_table_r000500.pkl", 2)
    write_table(work_dir / "q_table.pkl", 3)

    monkeypatch.setattr(
        train_ql,
        "benchmark_fast",
        lambda config, progress=False: {
            "per_agent": {
                "attackontensor_ql_0": {"score_mean": 50.0, "survival_rate": 1.0}
            }
        },
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    best = train_ql.select_best_checkpoint(args_for(), get_stage(2), run_dir, work_dir)

    assert best == work_dir / "q_table.pkl", (
        f"tie went to {best.name}; the run's final table carries the most training"
    )


def test_training_flags_reach_the_agents_config(tmp_path):
    """Every knob the CLI exposes must actually arrive in the subprocess.

    `alpha_decay` was implemented at train.py:184 and documented in Phase 0 as
    the fix for constant-alpha Q-values tracking a moving target -- but it had
    no CLI flag and `stage_environment` never exported it, so every run for two
    phases silently trained with constant alpha. A flag that is plumbed but not
    exported fails exactly this way: silently, with plausible results.
    """
    args = train_ql.build_parser().parse_args(
        ["--stage", "2", "--alpha", "0.2", "--alpha-decay", "0.01",
         "--alpha-min", "0.005", "--seed", "3"]
    )
    stage = get_stage(2)
    env = train_ql.stage_environment(args, stage, tmp_path, tmp_path)

    assert env["AOT_QL_ALPHA"] == "0.2"
    assert env["AOT_QL_ALPHA_DECAY"] == "0.01"
    assert env["AOT_QL_ALPHA_MIN"] == "0.005"
    assert "AOT_QL_SEED" in env

    # And the agent must parse them back to the right types.
    import os

    from agent_code.attackontensor_ql.config import QLConfig

    keep = {k: os.environ.get(k) for k in
            ("AOT_QL_ALPHA", "AOT_QL_ALPHA_DECAY", "AOT_QL_ALPHA_MIN")}
    try:
        for key in keep:
            os.environ[key] = env[key]
        config = QLConfig.load()
        assert config.alpha == pytest.approx(0.2)
        assert config.alpha_decay == pytest.approx(0.01)
        assert config.alpha_min == pytest.approx(0.005)
    finally:
        for key, value in keep.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_alpha_decay_defaults_to_off():
    """Constant alpha stays the default: turning the schedule on changes what
    every existing result means, so it must be an explicit choice."""
    args = train_ql.build_parser().parse_args(["--stage", "2"])
    assert args.alpha_decay == 0.0


def test_checkpoint_spacing_is_configurable(tmp_path):
    """Selection can only promote a snapshot that was taken.

    Stage 2 measured adjacent 250-round snapshots differing by 20+ coins, so
    spacing bounds how good a table selection can find. Stages 3-4 are 2-3x
    longer and want finer spacing; the default stays 250 so existing results
    remain reproducible.
    """
    stage = get_stage(2)
    default = train_ql.build_parser().parse_args(["--stage", "2"])
    assert default.checkpoint_every == 250
    assert train_ql.stage_environment(
        default, stage, tmp_path, tmp_path)["AOT_QL_CHECKPOINT_EVERY"] == "250"

    finer = train_ql.build_parser().parse_args(
        ["--stage", "3", "--checkpoint-every", "150"])
    assert train_ql.stage_environment(
        finer, get_stage(3), tmp_path, tmp_path)["AOT_QL_CHECKPOINT_EVERY"] == "150"

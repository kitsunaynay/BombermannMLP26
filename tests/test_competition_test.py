"""Tests for the submission-selection read-out."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from competition_test import lineups, parse_candidate, summarise  # noqa: E402


def game(match, seed, scores, failed=None, log_problems=None):
    agents = {name: {"score": float(score), "coins": 0.0, "kills": 0.0, "suicides": 0.0,
                     "steps": 100.0, "time": 0.5, "invalid": 0.0}
              for name, score in scores.items()}
    return {"match": match, "table": list(scores), "seed": seed, "failed": failed,
            "agents": agents, "overruns": {}, "log_problems": log_problems or {}}


def test_plain_directory_is_its_own_name():
    candidate = parse_candidate("attackontensor_ppo")
    assert (candidate.name, candidate.agent_dir, candidate.checkpoint) == \
        ("attackontensor_ppo", "attackontensor_ppo", None)


def test_checkpoint_variant_gets_an_importable_name():
    candidate = parse_candidate("attackontensor_ppo@checkpoints/ppo-ft2-rb3-robust-s1.pt")
    assert candidate.name == "attackontensor_ppo__ppo_ft2_rb3_robust_s1"
    assert candidate.name.isidentifier()
    assert candidate.checkpoint.name == "ppo-ft2-rb3-robust-s1.pt"
    assert parse_candidate("s1=attackontensor_ppo@x.pt").name == "attackontensor_ppo__s1"


def test_label_without_checkpoint_is_rejected():
    with pytest.raises(ValueError):
        parse_candidate("s1=attackontensor_ppo")


def test_every_table_seats_four():
    for count in (2, 3, 4, 5):
        names = [f"agent{i}" for i in range(count)]
        tables = lineups(names, ("baseline", "mixed", "rivals"))
        assert all(len(table) == 4 for match in tables.values() for table in match)
        assert len(tables["baseline"]) == count
        assert all(any(name in table for table in tables["rivals"]) for name in names)
    assert "rivals" not in lineups(["alone"], ("baseline", "rivals"))


def test_clear_lead_is_found_and_paired():
    records = []
    for seed in range(40):
        arena = seed % 5          # shared arena luck, removed by pairing
        records.append(game("baseline", seed, {"a": 8 + arena, "rule_based_agent_0": 1}))
        records.append(game("baseline", seed, {"b": 6 + arena, "rule_based_agent_0": 1}))
    verdict = summarise(records, ["a", "b"])["verdict"]
    assert verdict["submit"] == "a" and verdict["clear"]
    assert verdict["lead"][0]["diff"] == pytest.approx(2.0)
    assert verdict["lead"][0]["ci"] == (pytest.approx(2.0), pytest.approx(2.0))


def test_small_lead_is_not_called_clear():
    records = []
    for seed in range(40):
        records.append(game("baseline", seed, {"a": 5.2 if seed % 2 else 5.0, "x": 0}))
        records.append(game("baseline", seed, {"b": 5.0, "x": 0}))
    verdict = summarise(records, ["a", "b"])["verdict"]
    assert verdict["submit"] == "a" and not verdict["clear"]


def test_crashed_candidate_is_never_recommended():
    records = []
    for seed in range(10):
        records.append(game("baseline", seed, {"strong": 12, "x": 0}))
        records.append(game("baseline", seed, {"weak": 3, "x": 0}))
    records.append(game("baseline", 99, {"strong": 0, "x": 0}, failed="traceback: boom"))
    result = summarise(records, ["strong", "weak"])
    assert not result["per_candidate"]["strong"]["eligible"]
    assert result["verdict"]["submit"] == "weak"

    logged = [game("baseline", 0, {"strong": 12, "x": 0}, log_problems={"strong": ["a.log: contains 'Traceback'"]}),
              game("baseline", 0, {"weak": 3, "x": 0})]
    assert summarise(logged, ["strong", "weak"])["verdict"]["submit"] == "weak"

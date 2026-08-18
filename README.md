```bash
# Pull from main
git fetch upstream
git checkout main
git merge upstream/main
```


# bomberman_rl — Team AttackOnTensor

Setup for a project/competition amongst students to train a winning Reinforcement
Learning agent for the classic game Bomberman.

This fork adds our two learning agents and the experiment infrastructure around
them. Full design rationale is in **[`docs/BLUEPRINT.md`](docs/BLUEPRINT.md)**;
commands to reproduce every number are in
**[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)**.

## The two agents

| | Model 1 | Model 2 |
|---|---|---|
| Directory | `agent_code/attackontensor_ql/` | `agent_code/attackontensor_ppo/` |
| Method | Feature-based n-step Double Q-learning | PPO actor-critic (CNN) |
| State | 9–12 engineered discrete features | `(13, 17, 17)` spatial tensor |
| Dependencies | numpy | numpy, torch |

Both are self-contained: everything they need at inference lives inside their own
directory, because the tournament copies exactly one directory into a clean
checkout of the framework.

## Quick start

```bash
pip install -r requirements.txt

# Watch an agent play
python main.py play --agents attackontensor_ql rule_based_agent

# Train the Q-learning agent through the curriculum (Tasks 1-4 of the brief)
python tools/train_ql.py --stage 1

# Train PPO with parallel rollout collection
python tools/train_ppo.py --stage 1 --workers 8 --total-steps 500000

# Compare against every provided baseline over 30 seeds
python tools/benchmark.py --agents attackontensor_ql rule_based_agent \
    coin_collector_agent random_agent --seeds 30

# Confirm we decide within the 0.5s tournament budget
python tools/latency_check.py --agent attackontensor_ppo

# Regenerate all report figures and tables
python tools/make_report_assets.py
```

## Repository layout

```
shared/kit/       game utilities — the ONLY editable copy
agent_code/       the two agents (each vendors shared/kit into kit/)
blib/             training + evaluation infrastructure (never submitted)
tools/            command-line entry points
tests/            pytest suite
docs/             blueprint and reproduction guide
```

`shared/kit/` is vendored into each agent by `tools/sync_kit.py`. **Edit
`shared/kit/`, never `agent_code/*/kit/`**, then re-run the tool;
`tools/sync_kit.py --check` runs as a unit test and fails if a copy has drifted.

## Tests

```bash
python -m pytest tests/ -q
```

Notable coverage: blast geometry cross-checked against the framework's own
`items.Bomb`, GAE checked against an independent reference implementation, and
`tests/test_fast_env_parity.py`, which drives the fast training environment and a
stock `BombeRLeWorld` through identical actions and asserts they evolve
identically step for step.

## Changes to framework files

Two upstream bugs are fixed for development convenience. Both files are replaced
by the graders' originals at tournament time, so nothing depends on them:

- `fallbacks.py` — `LOADED_PYGAME` was `True` even when the import failed, which
  made `main.py`'s guard unreachable.
- `main.py` — added `--fps`; `GUI.make_video` read `args.fps`, which no argument
  ever defined, so `--make-video` always raised `AttributeError`.

`agent_code/scripted_test_agent/` is a deterministic test fixture used by the
parity test, not a competitor.

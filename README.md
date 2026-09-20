# bomberman_rl — Team AttackOnTensor

the agents implemented:

| | model 1 | Model 2 |
|---|---|---|
| Directory | `agent_code/attackontensor_ql/` | `agent_code/attackontensor_ppo/` |
| Method | Feature-based n-step Double Q-learning | PPO actor-critic (CNN) |
| State | 9–12 engineered discrete features | `(13, 17, 17)` spatial tensor |


more details in the report.

## Quick stuff

```bash
# watch agent
python main.py play --agents attackontensor_ql rule_based_agent

# train ql
python tools/train_ql.py --stage 1

# train ppo
python tools/train_ppo.py --stage 1 --workers 8 --total-steps 500000

# comparison on 30 seeds, vs baseline agents
python tools/benchmark.py --agents attackontensor_ql rule_based_agent \
    coin_collector_agent random_agent --seeds 30

# 0.5s tournament requirement
python tools/latency_check.py --agent attackontensor_ppo

# figs and table update
python tools/make_report_assets.py
```


### Ai Acknowledgment:
AI was used in the following way:
1. Code was written on our own, restructured, commented, and optimized by AI for readability and clean repository that all members can understand each others code more easily.
2. Recommended training harness was implemented by human and optimized by AI for high quality code and optimal training. Double checked by human.
3. Base repository structure was created with AI to start clean and organized. Made with detailed human instructions and double checks. No agent implementations yet.
4. Support with a training feedback loop to automate testing and promotion. Referred to as "curriculum" in the files. Curriculum given by human.


### Base repo changes:

Four upstream bugs are fixed for development convenience:

- `fallbacks.py` — `LOADED_PYGAME` was `True` even when the import failed, which
  made `main.py`'s guard unreachable.
- `main.py` — added `--fps`; `GUI.make_video` read `args.fps`, which no argument
  ever defined, so `--make-video` always raised `AttributeError`.
- `settings.py` — agent logging defaulted to `DEBUG`, which cost 2.1 GB over one
  overnight sweep. It is now silent by default and opt-in per invocation:

  ```bash
  AOT_LOG_LEVEL=DEBUG python main.py play --agents attackontensor_ql
  ```

- `agents.py` — `settings.LOG_MAX_FILE_SIZE` was declared and never used, so a
  `DEBUG` log grew without bound. The handler now rotates at that limit.

Both training and `--backend main` benchmarking shell out to `main.py`, so the
logging path is on the hot path for every long job here. Note that the log file
name depends only on the agent's name: parallel runs share it, so `AOT_LOG_LEVEL`
during a sweep produces interleaved, not per-run, logs.

`agent_code/scripted_test_agent/` is a deterministic test fixture used by the
parity test, not a competitor.

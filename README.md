##### COPYRIGHT NOTICE
Parts of this repository belong to the original course's repository supplied by Prof Köthe (https://github.com/ukoethe/bomberman_rl) and are a copy from August 2026's main branch.

# Team AttackOnTensor

The agents implemented:

| | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| Directory | `agent_code/attackontensor_ql/` | `agent_code/attackontensor_ppo/` | `agent_code/survival_linear_ppo_v4/` |
| Method | Feature-based n-step Double Q-learning | PPO actor-critic (CNN) | Linear PPO actor-critic with engineered features and safety constraints |
| State | 9–12 engineered discrete features | `(13, 17, 17)` spatial tensor | Engineered state and action features |

More details in the report.

The training and controlled experiments for the linear PPO agent are documented in:

`notebooks/survival_linear_ppo.ipynb`

## Quick stuff

```bash
# watch Q-learning agent
python main.py play --agents attackontensor_ql rule_based_agent

# watch CNN PPO agent
python main.py play --agents attackontensor_ppo rule_based_agent

# watch linear PPO agent
python main.py play --agents survival_linear_ppo_v4 rule_based_agent

# train ql
python tools/train_ql.py --stage 1

# train CNN ppo
python tools/train_ppo.py --stage 1 --workers 8 --total-steps 500000

# linear PPO training and experiments
# see notebooks/survival_linear_ppo.ipynb
# training callbacks are in agent_code/survival_linear_ppo_v4/train.py

# comparison on 30 seeds, vs baseline agents
python tools/benchmark.py --agents attackontensor_ql rule_based_agent \
    coin_collector_agent random_agent --seeds 30

# 0.5s tournament requirement
python tools/latency_check.py --agent attackontensor_ppo

# figs and table update
python tools/make_report_assets.py

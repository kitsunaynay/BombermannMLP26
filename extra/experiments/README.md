# Experiment results

This folder contains results from the additional experiments.

The experiments focus on two questions:

1. Whether adding explicit survival-related features improves the CNN PPO agent.
2. How Double DQN compares with PPO when using the same spatial representation,
   reward design, safety mechanism, training environment, and similar training setup.

The experiments include multiple independent training seeds, checkpoint
selection on held-out validation seeds, and evaluation on matched test seeds.

The final PPO vs. Double DQN evaluation used six independently trained agents
for each algorithm and 50 matched test seeds per agent.

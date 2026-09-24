# Additional agents

This folder contains the additional agents used for the experiments.

- `beta_control`: CNN PPO agent using the standard 13-channel spatial representation.
- `beta_survival`: CNN PPO variant with four additional survival-related input channels.
- `beta_dqn`: CNN Double DQN agent using the same 13-channel spatial representation as the PPO control agent.

The control and survival agents were used to test whether the additional
survival features improved performance. The DQN agent was used for the
controlled comparison between PPO and Double DQN.

# Saved model checkpoints

The repository retains the best-held-out checkpoint for each algorithm based on the recorded mean evaluation reward.

| Algorithm | Checkpoint | Run | Mean reward | Standard deviation |
|---|---|---:|---:|---:|
| DQN | `dqn/dqn_run_3.pt` | 3 | 4.38901 | 0.58499 |
| REINFORCE | `pg/reinforce_run_8.pt` | 8 | 1.28737 | 0.91459 |
| PPO | `pg/ppo_run_2.pt` | 2 | 2.66589 | 0.60703 |
| A2C | `pg/a2c_run_3.pt` | 3 | 1.68748 | 0.63561 |

The complete 10-run evidence for each algorithm remains in `logs/*_experiments.csv` and `logs/*_experiments.json`. Run 9001 smoke-test checkpoints are intentionally excluded.

## Playback architecture

- DQN: `QNetwork(hidden_size=64)`
- REINFORCE: `PolicyNet(hidden_size=64)`
- PPO: `ActorCriticNet(hidden_size=64)`
- A2C: `ActorCriticNet(hidden_size=128)`

These match the configurations used for the listed winning runs.

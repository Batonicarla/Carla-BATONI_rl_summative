# Saved model checkpoints

The repository bundles one validation-selected checkpoint per algorithm for demo playback and
independent evaluation. Logs retain the results for all 40 configurations. The demo's default policy is the
checkpoint with the highest validation mean across the four algorithms.

| Algorithm | Checkpoint | Run | Validation mean | SD |
|---|---|---:|---:|---:|
| DQN | [dqn_run_2.pt](dqn/dqn_run_2.pt) | 2 | +1.094 | 0.279 |
| REINFORCE | [reinforce_run_8.pt](pg/reinforce_run_8.pt) | 8 | -1.091 | 0.283 |
| PPO | [ppo_run_8.pt](pg/ppo_run_8.pt) | 8 | -0.344 | 0.294 |
| A2C | [a2c_run_8.pt](pg/a2c_run_8.pt) | 8 | +0.374 | 0.255 |

Network widths are recorded in the configuration for each run. See the
[full run table](../assets/all_runs_table.csv) and [independent test results](../logs/test_evaluation.csv).

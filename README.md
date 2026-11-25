# ElderCare AI

ElderCare AI compares DQN, REINFORCE, PPO, and Advantage Actor-Critic (A2C) in a custom Gymnasium environment. It models an assistant that recommends wellbeing activities to fictional older adults and takes account of preferences, mobility, fatigue, and declined suggestions.

> My personal mission as a software engineer is to empower old age wellbeing specifically, through the creation of human-centred digital solutions and community-oriented systems, which will enable older adults to experience more fulfilling, connected, and healthy lives.

![ElderCare AI simulation dashboard](assets/environment_demo.png)

## Run the project

From the cloned repository root, with `uv` installed:

```text
uv sync --locked
uv run main.py test
uv run main.py smoke
uv run main.py evaluate
```

The lockfile selects dependencies for Python 3.12.

### Run the Pygame demo

```bash
uv run main.py demo --algorithm best --seed 2026 --step-seconds 1.25
```

The demo loads the checkpoint with the highest validation mean and runs the episode automatically. The terminal prints each recommendation, the participant's response, the step reward, the cumulative reward, and the average wellbeing.

When the episode ends, the terminal prints the reason and **Pygame keeps the final screen open**. You can explain the final reward and wellbeing indicators, then click the window's **X** to exit. Leave `--hold-seconds` out when recording. Add it only when you want the window to close automatically after a set number of seconds.

To view a different policy, replace `best` with `dqn`, `reinforce`, `ppo`, `a2c`, or `random`.

## Environment

Each episode starts on day one at 08:00. A seeded reset samples a profile and seven wellbeing indicators between 0.45 and 0.75. The simulation uses 12 decision slots per day for seven days.

| Action | Real-world equivalent |
|---|---|
| 0. No intervention | Give the person space |
| 1. Hydration | Offer an accessible hydration reminder |
| 2. Suitable physical activity | Suggest movement suited to mobility |
| 3. Social connection | Suggest a call, conversation, or check-in |
| 4. Rest | Suggest a break |
| 5. Mental stimulation | Suggest a puzzle, reading, or music |
| 6. Accessible community activity | Suggest a community activity suited to the person |
| 7. Daily routine assistance | Offer support with a manageable daily task |

The 18 observations are normalized to [0, 1]. They include seven wellbeing indicators (energy, social connection, physical activity, hydration, mood, cognitive engagement, and routine completion), time of day, fatigue, an eight-value one-hot previous action, and elapsed week progress. Profile preferences and scheduled delayed effects influence transitions. These are latent inputs to the simulator.

The reward is five times the weighted wellbeing change, plus a small action-fit bonus, minus penalties for unsuitable, repeated, and declined recommendations. No intervention has its own context-dependent bonus. A declined recommendation has no direct action effect. Accepted recommendations may change indicators immediately and schedule a benefit for the following decision. Natural changes apply each step.

An episode terminates after three consecutive steps with at least three wellbeing indicators below 0.18. It otherwise truncates at 84 steps. The Pygame/SDL2 scene highlights the recommended station in an isometric wellbeing centre. The participant moves there after an accepted prompt and stays put after a decline. The renderer prefers the OpenGL presentation backend and can use the platform's fallback backend. The environment also provides an offscreen RGB frame and text output.

## Algorithms and experiments

| Algorithm | Training method |
|---|---|
| DQN | TorchRL replay buffer, epsilon-greedy exploration, and target-network Bellman updates |
| REINFORCE | TorchRL categorical log-probabilities and entropy and Monte Carlo policy gradients |
| PPO | TorchRL categorical log-probabilities and entropy, generalized advantage estimation, and a clipped probability-ratio objective |
| A2C | TorchRL categorical log-probabilities and entropy and n-step, bootstrapped actor-critic updates |

All the learners use the same Gymnasium environment. The TorchRL smoke check also exercises the Gymnasium wrapper. The logged sweep uses 10 distinct configurations per algorithm and 300 episodes per configuration. Greedy rollouts on 20 common validation seeds determine the selected run per algorithm. The independent test compares those saved policies with a uniform random agent over 50 additional seeds.

```text
uv run main.py train --episodes 300 --runs-per-algorithm 10 --seed 42
uv run main.py plots
uv run main.py evaluate
```

- [All 40 configurations and validation results](assets/all_runs_table.csv)
- [Selected runs](logs/analysis_summary.csv)
- [Independent test results](logs/test_evaluation.csv)
- [Sampled-policy and heuristic diagnostics](logs/evaluation_diagnostics.csv)
- [Training reward curves](assets/reward_comparison.png)
- [Training stability](assets/stability_comparison.png)
- [Convergence curves](assets/convergence_comparison.png)

### Baselines and policy behaviour

On the same 50 test seeds, DQN averaged **+1.03**, uniform random **-1.04**, and the immediate-fit heuristic **-0.81**. Adding a rule that excludes the previous action raised the heuristic to **+1.57**, ahead of every trained policy tested here. Both heuristic variants use the simulator's latent profile preferences. The rule explicitly avoids triggering the repetition penalty, so this comparison shows how much a simple policy can accomplish in this reward design. With no prompts, all 50 episodes ended early.

Inspection of archived checkpoints confirmed that greedy REINFORCE runs 1 and 7 chose hydration on all 4,200 test decisions. Run 10 chose hydration on 99.86% of decisions. Always choosing hydration scored **-11.57**. PPO run 1 and A2C run 1 picked exactly the same greedy action sequences on all 50 test seeds. Both checkpoints' weights differed from their seeded initializations, so matching scores alone cannot establish that training failed to update them. [Checkpoint audit](logs/policy_behavior_audit.csv)

### Longer training follow-up

Each policy-gradient algorithm was retrained from scratch for 600 episodes using its original validation-selected configuration and training seed. This doubles the interaction budget to 50,400 per run. The table shows mean returns on the same 50 test seeds. The follow-up is a single-seed budget comparison.

| Algorithm | 300 greedy | 600 greedy | 300 sampled | 600 sampled |
|---|---:|---:|---:|---:|
| [REINFORCE](logs/reinforce_budget_600.csv) | -1.10 | -0.56 | -0.67 | -0.06 |
| [PPO](logs/ppo_budget_600.csv) | -0.44 | +0.10 | +0.45 | +0.64 |
| [A2C](logs/a2c_budget_600.csv) | +0.38 | -0.06 | -0.03 | +0.15 |

Longer training improved both inference modes for REINFORCE and PPO. A2C improved when sampling actions and declined under greedy evaluation. Reproduce a follow-up with `uv run python -m training.budget_study ppo --episodes 600`, substituting `reinforce` or `a2c` as needed. The checked-in demo checkpoints still come from the original validation sweep.

The convergence criterion requires 50 consecutive values of the 20-episode moving average within 5% of the final 20-episode mean. The original selected runs and the three follow-ups did not meet it within their respective budgets. The CSV records this explicitly as `not_reached`.

The simulator uses synthetic profiles, scores, and response probabilities. The findings apply to this simulated environment. Each configuration has one training seed and can vary multiple settings at once, so the sweep supports comparisons between configurations.

## Repository layout

- `environment/` contains the Gymnasium environment and rendering.
- `training/` contains the four learners and analysis tools.
- `models/` contains saved validation-selected policies.
- `logs/` and `assets/` contain experiment evidence.
- `tests/` checks environment, rendering, TorchRL and training behaviour.

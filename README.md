# ElderCare AI — RL Summative Project

A reproducible ElderCare Gymnasium environment comparing DQN, REINFORCE, PPO, and synchronous A2C using PyTorch and TorchRL.

> Educational synthetic simulation; not a medical decision-making system.

## Run

```powershell
uv sync
uv run main.py test
uv run main.py smoke
uv run main.py demo --algorithm random --hold-seconds 60
uv run main.py demo --algorithm dqn --hold-seconds 60
uv run main.py plots
```

The DQN demo loads `models/dqn/dqn_run_2.pt`, the retained best-held-out DQN checkpoint.

## Full experiment reproduction

```powershell
uv run main.py train --episodes 300 --runs-per-algorithm 10 --seed 42
uv run main.py evaluate
uv run main.py plots
```

## Repository contents

- `environment/`: custom Gymnasium environment and Pygame rendering
- `training/`: DQN, policy-gradient algorithms, and plot analysis
- `tests/`: environment and TorchRL compatibility tests
- `main.py`: test, smoke, train, evaluate, plots, and demo CLI
- `logs/`: 40-run CSV/JSON experiment evidence
- `assets/`: reward, evaluation, and loss plots
- `models/dqn/dqn_run_2.pt`: retained best DQN checkpoint
- `pyproject.toml`, `uv.lock`: reproducible uv dependency setup

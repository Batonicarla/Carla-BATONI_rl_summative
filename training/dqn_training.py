"""DQN training and hyperparameter sweeps for ElderCareEnv."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer

from environment.custom_env import ElderCareEnv, OBSERVATION_NAMES

OBSERVATION_DIM = len(OBSERVATION_NAMES)
EVALUATION_SEEDS = tuple(range(100_000, 100_020))


class QNetwork(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 8),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


def evaluate_dqn(policy: QNetwork, seeds: tuple[int, ...] = EVALUATION_SEEDS) -> tuple[float, float]:
    rewards = []
    policy.eval()
    for seed in seeds:
        env = ElderCareEnv()
        obs, _ = env.reset(seed=seed)
        total = 0.0
        while True:
            with torch.no_grad():
                action = int(policy(torch.tensor(obs, dtype=torch.float32)).argmax().item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


def run_dqn_experiment(run_id: int, episodes: int, seed: int, config: dict) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = ElderCareEnv()
    env.action_space.seed(seed)
    policy = QNetwork(config["hidden_size"])
    target = QNetwork(config["hidden_size"])
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["learning_rate"])
    replay = ReplayBuffer(storage=LazyTensorStorage(config["replay_capacity"]), batch_size=config["batch_size"])

    episode_rewards = []
    episode_lengths = []
    losses = []
    epsilon = config["epsilon_start"]
    interaction_count = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0

        for step in range(env.max_steps):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(policy(torch.tensor(obs, dtype=torch.float32)).argmax().item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            replay.add(TensorDict({
                "state": torch.from_numpy(obs),
                "action": torch.tensor(action, dtype=torch.long),
                "reward": torch.tensor(reward, dtype=torch.float32),
                "next_state": torch.from_numpy(next_obs),
                "done": torch.tensor(terminated or truncated, dtype=torch.float32),
            }, batch_size=[]))
            obs = next_obs
            total += reward

            if len(replay) >= config["batch_size"] and interaction_count % 4 == 0:
                batch = replay.sample()
                states = batch["state"]
                actions = batch["action"]
                rewards = batch["reward"]
                next_states = batch["next_state"]
                dones = batch["done"]

                q_values = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_values = target(next_states).max(1).values
                    targets = rewards + config["gamma"] * next_values * (1.0 - dones)
                loss = nn.functional.smooth_l1_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step()
                losses.append(float(loss.item()))

            interaction_count += 1
            epsilon = max(config["epsilon_end"], epsilon * config["epsilon_decay"])
            if interaction_count % config["target_update"] == 0:
                target.load_state_dict(policy.state_dict())

            if terminated or truncated:
                break

        episode_rewards.append(total)
        episode_lengths.append(step + 1)

    mean_reward, std_reward = evaluate_dqn(policy, EVALUATION_SEEDS)
    result = {
        "algorithm": "dqn", "run_id": run_id, "seed": seed,
        "config": config, "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths, "training_losses": losses,
        "evaluation_mean_reward": mean_reward, "evaluation_std_reward": std_reward,
    }

    output = Path("logs")
    output.mkdir(exist_ok=True)
    with (output / "dqn_experiments.json").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")

    checkpoint_dir = Path("models/dqn")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), checkpoint_dir / f"dqn_run_{run_id}.pt")
    return result


def run_dqn_experiments(episodes: int, run_count: int, seed: int) -> list[dict]:
    configurations = [
        {"learning_rate": 2.5e-4, "gamma": 0.95, "batch_size": 32, "replay_capacity": 5000, "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9990, "target_update": 250, "hidden_size": 64},
        {"learning_rate": 5e-4, "gamma": 0.97, "batch_size": 64, "replay_capacity": 10000, "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9995, "target_update": 500, "hidden_size": 64},
        {"learning_rate": 1e-3, "gamma": 0.99, "batch_size": 128, "replay_capacity": 20000, "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9998, "target_update": 1000, "hidden_size": 64},
        {"learning_rate": 2e-3, "gamma": 0.995, "batch_size": 32, "replay_capacity": 5000, "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9999, "target_update": 2000, "hidden_size": 64},
        {"learning_rate": 4e-3, "gamma": 0.95, "batch_size": 64, "replay_capacity": 10000, "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9990, "target_update": 250, "hidden_size": 64},
        {"learning_rate": 2.5e-4, "gamma": 0.97, "batch_size": 128, "replay_capacity": 20000, "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9995, "target_update": 500, "hidden_size": 64},
        {"learning_rate": 5e-4, "gamma": 0.99, "batch_size": 32, "replay_capacity": 5000, "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9998, "target_update": 1000, "hidden_size": 64},
        {"learning_rate": 1e-3, "gamma": 0.995, "batch_size": 64, "replay_capacity": 10000, "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9999, "target_update": 2000, "hidden_size": 64},
        {"learning_rate": 2e-3, "gamma": 0.95, "batch_size": 128, "replay_capacity": 20000, "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9992, "target_update": 500, "hidden_size": 64},
        {"learning_rate": 4e-3, "gamma": 0.99, "batch_size": 64, "replay_capacity": 10000, "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9997, "target_update": 1000, "hidden_size": 64},
    ]
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    (log_dir / "dqn_experiments.json").write_text("", encoding="utf-8")

    runs = []
    for run_id in range(1, min(run_count, len(configurations)) + 1):
        config = dict(configurations[run_id - 1])
        result = run_dqn_experiment(run_id, episodes, seed + run_id * 100, config)
        runs.append(result)
        print(f"[DQN {run_id}/{run_count}] mean={result['evaluation_mean_reward']:.3f}", flush=True)

    log = Path("logs/dqn_experiments.csv")
    with log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "seed", "config", "evaluation_mean_reward", "evaluation_std_reward"])
        writer.writeheader()
        for result in runs:
            writer.writerow({
                "run_id": result["run_id"], "seed": result["seed"],
                "config": json.dumps(result["config"], sort_keys=True),
                "evaluation_mean_reward": result["evaluation_mean_reward"],
                "evaluation_std_reward": result["evaluation_std_reward"],
            })
    return runs

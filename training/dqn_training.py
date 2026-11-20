"""DQN training and hyperparameter sweeps for ElderCareEnv."""

from __future__ import annotations

import csv
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from environment.custom_env import ElderCareEnv


class QNetwork(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 8),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


def evaluate_dqn(policy: QNetwork, seeds: list[int], episodes_per_seed: int = 20) -> tuple[float, float]:
    rewards = []
    policy.eval()
    for seed in seeds:
        for episode in range(episodes_per_seed):
            env = ElderCareEnv()
            obs, _ = env.reset(seed=seed * 1000 + episode)
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
    policy = QNetwork(config["hidden_size"])
    target = QNetwork(config["hidden_size"])
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["learning_rate"])
    replay = deque(maxlen=config["replay_capacity"])

    episode_rewards = []
    episode_lengths = []
    losses = []
    epsilon = config["epsilon_start"]

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0

        for step in range(12):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(policy(torch.tensor(obs, dtype=torch.float32)).argmax().item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            replay.append((obs.copy(), action, float(reward), next_obs.copy(), terminated or truncated))
            obs = next_obs
            total += reward

            if len(replay) >= config["batch_size"]:
                batch = random.sample(replay, config["batch_size"])
                states = torch.tensor(np.stack([x[0] for x in batch]), dtype=torch.float32)
                actions = torch.tensor([x[1] for x in batch], dtype=torch.long)
                rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
                next_states = torch.tensor(np.stack([x[3] for x in batch]), dtype=torch.float32)
                dones = torch.tensor([x[4] for x in batch], dtype=torch.float32)

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

            if terminated or truncated:
                break

        epsilon = max(config["epsilon_end"], epsilon * config["epsilon_decay"])
        if episode % config["target_update"] == 0:
            target.load_state_dict(policy.state_dict())

        episode_rewards.append(total)
        episode_lengths.append(step + 1)

    mean_reward, std_reward = evaluate_dqn(policy, [seed + 10_000, seed + 20_000])
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
    base = {
        "learning_rate": 1e-3, "gamma": 0.99, "batch_size": 64,
        "replay_capacity": 10_000, "epsilon_start": 1.0,
        "epsilon_end": 0.05, "epsilon_decay": 0.995,
        "target_update": 100, "hidden_size": 64,
    }
    runs = []
    for run_id in range(1, run_count + 1):
        config = dict(base)
        config["learning_rate"] *= [0.25, 0.5, 1.0, 2.0, 4.0][(run_id - 1) % 5]
        config["gamma"] = [0.95, 0.97, 0.99, 0.995][(run_id - 1) % 4]
        config["batch_size"] = [32, 64, 128][(run_id - 1) % 3]
        config["replay_capacity"] = [2_000, 5_000, 10_000, 20_000][(run_id - 1) % 4]
        config["epsilon_decay"] = [0.97, 0.985, 0.995, 0.999][(run_id - 1) % 4]
        config["target_update"] = [25, 50, 100, 200][(run_id - 1) % 4]
        result = run_dqn_experiment(run_id, episodes, seed + run_id * 100, config)
        runs.append(result)
        print(f"[DQN {run_id}/{run_count}] mean={result['evaluation_mean_reward']:.3f}")

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

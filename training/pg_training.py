"""Policy-gradient training: REINFORCE, PPO, and synchronous A2C."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchrl.modules import MaskedCategorical

from environment.custom_env import ElderCareEnv, OBSERVATION_NAMES

OBSERVATION_DIM = len(OBSERVATION_NAMES)
EVALUATION_SEEDS = tuple(range(100_000, 100_020))


class PolicyNet(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBSERVATION_DIM, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 8))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class ActorCriticNet(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(OBSERVATION_DIM, hidden_size), nn.ReLU())
        self.actor = nn.Linear(hidden_size, 8)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(observation)
        return self.actor(features), self.critic(features).squeeze(-1)


def action_distribution(logits: torch.Tensor) -> MaskedCategorical:
    return MaskedCategorical(
        logits=logits, mask=torch.ones_like(logits, dtype=torch.bool), use_cross_entropy=False
    )


def sample_action(logits: torch.Tensor) -> int:
    return int(torch.multinomial(logits.softmax(dim=-1), num_samples=1).item())


def discount_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    returns = torch.zeros(len(rewards), dtype=torch.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def compute_gae(rewards: list[float], values: torch.Tensor, gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute λ-return targets and generalized advantage estimates."""
    advantages = torch.zeros(len(rewards), dtype=torch.float32)
    running_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = 0.0 if index == len(rewards) - 1 else values[index + 1].item()
        delta = rewards[index] + gamma * next_value - values[index].item()
        running_advantage = delta + gamma * gae_lambda * running_advantage
        advantages[index] = running_advantage
    return advantages + values.detach(), advantages


def evaluate_policy(policy, seeds: tuple[int, ...] = EVALUATION_SEEDS) -> tuple[float, float]:
    rewards = []
    policy.eval()
    for seed in seeds:
        env = ElderCareEnv()
        obs, _ = env.reset(seed=seed)
        total = 0.0
        while True:
            state = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                output = policy(state)
                logits = output[0] if isinstance(output, tuple) else output
                action = int(logits.argmax().item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


def train_reinforce(episodes: int, seed: int, config: dict) -> tuple[nn.Module, list[float], list[float], list[float], list[float]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = ElderCareEnv()
    env.action_space.seed(seed)
    policy = PolicyNet(config["hidden_size"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["learning_rate"])
    episode_rewards = []
    losses = []
    entropies = []
    episode_lengths = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        states, actions, rewards = [], [], []
        total = 0.0

        for step in range(env.max_steps):
            state = torch.from_numpy(obs)
            with torch.no_grad():
                action = sample_action(policy(state))
            states.append(state)
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            total += reward
            if terminated or truncated:
                break

        returns = discount_returns(rewards, config["gamma"])
        if config["normalize_returns"] and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        dist = action_distribution(policy(torch.stack(states)))
        policy_loss = -(dist.log_prob(torch.tensor(actions)) * returns).sum()
        entropy = dist.entropy().mean()
        loss = policy_loss - config["entropy_coefficient"] * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()

        episode_rewards.append(total)
        episode_lengths.append(step + 1)
        losses.append(float(loss.item()))
        entropies.append(float(entropy.detach().item()))

    return policy, episode_rewards, losses, entropies, episode_lengths


def train_ppo(episodes: int, seed: int, config: dict) -> tuple[nn.Module, list[float], list[float], list[float], list[float]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = ElderCareEnv()
    env.action_space.seed(seed)
    model = ActorCriticNet(config["hidden_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    episode_rewards = []
    losses = []
    entropies = []
    episode_lengths = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        states, actions, rewards = [], [], []
        total = 0.0

        for step in range(env.max_steps):
            state = torch.from_numpy(obs)
            with torch.no_grad():
                logits, _ = model(state)
                action = sample_action(logits)
            states.append(state)
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            total += reward
            if terminated or truncated:
                break

        state_tensor = torch.stack(states)
        action_tensor = torch.tensor(actions, dtype=torch.long)
        with torch.no_grad():
            old_logits, old_values = model(state_tensor)
            old_dist = action_distribution(old_logits)
            old_log_probs = old_dist.log_prob(action_tensor)
            entropies.append(float(old_dist.entropy().mean().item()))
        returns, advantages = compute_gae(
            rewards,
            old_values,
            config["gamma"],
            config["gae_lambda"],
        )

        if config["normalize_advantages"]:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        for _ in range(config["epochs"]):
            logits, predicted_values = model(state_tensor)
            dist = action_distribution(logits)
            new_log_probs = dist.log_prob(action_tensor)
            ratio = (new_log_probs - old_log_probs).exp()
            clipped = torch.clamp(ratio, 1.0 - config["clip_epsilon"], 1.0 + config["clip_epsilon"])
            actor_loss = -(torch.min(ratio * advantages, clipped * advantages)).mean()
            critic_loss = nn.functional.mse_loss(predicted_values, returns)
            entropy = dist.entropy().mean()
            loss = actor_loss + config["value_coefficient"] * critic_loss - config["entropy_coefficient"] * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.item()))

        episode_rewards.append(total)
        episode_lengths.append(step + 1)

    return model, episode_rewards, losses, entropies, episode_lengths


def train_a2c(episodes: int, seed: int, config: dict) -> tuple[nn.Module, list[float], list[float], list[float], list[float]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = ElderCareEnv()
    env.action_space.seed(seed)
    model = ActorCriticNet(config["hidden_size"])
    optimizer = torch.optim.Adam([
        {"params": list(model.shared.parameters()) + list(model.actor.parameters()), "lr": config["actor_learning_rate"]},
        {"params": list(model.critic.parameters()), "lr": config["critic_learning_rate"]},
    ])
    episode_rewards = []
    losses = []
    entropies = []
    episode_lengths = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        episode_length = 0
        done = False

        while not done:
            states, actions, rewards = [], [], []
            for _ in range(config["rollout_length"]):
                state = torch.from_numpy(obs)
                with torch.no_grad():
                    logits, _ = model(state)
                    action = sample_action(logits)
                states.append(state)
                actions.append(action)
                obs, reward, terminated, truncated, _ = env.step(action)
                rewards.append(float(reward))
                total += reward
                episode_length += 1
                done = terminated or truncated
                if done:
                    break

            if done:
                bootstrap = torch.zeros((), dtype=torch.float32)
            else:
                with torch.no_grad():
                    _, bootstrap = model(torch.tensor(obs, dtype=torch.float32))

            returns = torch.zeros(len(rewards), dtype=torch.float32)
            running_return = bootstrap.detach()
            for index in range(len(rewards) - 1, -1, -1):
                running_return = rewards[index] + config["gamma"] * running_return
                returns[index] = running_return

            state_tensor = torch.stack(states)
            action_tensor = torch.tensor(actions, dtype=torch.long)
            logits, predicted_values = model(state_tensor)
            advantages = returns - predicted_values.detach()
            dist = action_distribution(logits)
            log_probs = dist.log_prob(action_tensor)
            entropy = dist.entropy().mean()
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = config["value_coefficient"] * nn.functional.mse_loss(predicted_values, returns)
            loss = actor_loss + critic_loss - config["entropy_coefficient"] * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.item()))
            entropies.append(float(entropy.detach().item()))

        episode_rewards.append(total)
        episode_lengths.append(episode_length)

    return model, episode_rewards, losses, entropies, episode_lengths


def save_pg_run(algorithm: str, run_id: int, seed: int, config: dict, result: tuple) -> dict:
    policy, rewards, losses, entropies, lengths = result
    mean_reward, std_reward = evaluate_policy(policy)
    output = {
        "algorithm": algorithm, "run_id": run_id, "seed": seed, "config": config,
        "episode_rewards": rewards, "training_losses": losses,
        "policy_entropy": entropies, "episode_lengths": lengths,
        "evaluation_mean_reward": mean_reward, "evaluation_std_reward": std_reward,
    }
    path = Path("logs")
    path.mkdir(exist_ok=True)
    with (path / f"{algorithm}_experiments.json").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(output) + "\n")

    model_dir = Path("models/pg")
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), model_dir / f"{algorithm}_run_{run_id}.pt")

    return output


def run_pg_experiments(algorithm: str, episodes: int, run_count: int, seed: int) -> list[dict]:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    (log_dir / f"{algorithm}_experiments.json").write_text("", encoding="utf-8")

    runs = []
    if algorithm == "reinforce":
        configurations = [
            {"learning_rate": 3e-4, "gamma": 0.95, "hidden_size": 32, "normalize_returns": False, "entropy_coefficient": 0.0},
            {"learning_rate": 5e-4, "gamma": 0.97, "hidden_size": 64, "normalize_returns": True, "entropy_coefficient": 0.01},
            {"learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 128, "normalize_returns": False, "entropy_coefficient": 0.02},
            {"learning_rate": 2e-3, "gamma": 0.995, "hidden_size": 32, "normalize_returns": True, "entropy_coefficient": 0.05},
            {"learning_rate": 3e-4, "gamma": 0.95, "hidden_size": 64, "normalize_returns": False, "entropy_coefficient": 0.0},
            {"learning_rate": 5e-4, "gamma": 0.97, "hidden_size": 128, "normalize_returns": True, "entropy_coefficient": 0.01},
            {"learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 32, "normalize_returns": False, "entropy_coefficient": 0.02},
            {"learning_rate": 2e-3, "gamma": 0.995, "hidden_size": 64, "normalize_returns": True, "entropy_coefficient": 0.05},
            {"learning_rate": 3e-4, "gamma": 0.95, "hidden_size": 128, "normalize_returns": False, "entropy_coefficient": 0.0},
            {"learning_rate": 5e-4, "gamma": 0.97, "hidden_size": 32, "normalize_returns": True, "entropy_coefficient": 0.01},
        ]
    elif algorithm == "ppo":
        configurations = [
            {"learning_rate": 3e-4, "gamma": 0.95, "clip_epsilon": 0.1, "gae_lambda": 0.90, "epochs": 3, "value_coefficient": 0.25, "entropy_coefficient": 0.0, "hidden_size": 32, "normalize_advantages": False},
            {"learning_rate": 5e-4, "gamma": 0.97, "clip_epsilon": 0.2, "gae_lambda": 0.95, "epochs": 5, "value_coefficient": 0.50, "entropy_coefficient": 0.01, "hidden_size": 64, "normalize_advantages": True},
            {"learning_rate": 1e-3, "gamma": 0.99, "clip_epsilon": 0.3, "gae_lambda": 0.98, "epochs": 10, "value_coefficient": 1.00, "entropy_coefficient": 0.02, "hidden_size": 128, "normalize_advantages": False},
            {"learning_rate": 2e-3, "gamma": 0.995, "clip_epsilon": 0.1, "gae_lambda": 0.90, "epochs": 3, "value_coefficient": 0.25, "entropy_coefficient": 0.05, "hidden_size": 32, "normalize_advantages": True},
            {"learning_rate": 3e-4, "gamma": 0.95, "clip_epsilon": 0.2, "gae_lambda": 0.95, "epochs": 5, "value_coefficient": 0.50, "entropy_coefficient": 0.0, "hidden_size": 64, "normalize_advantages": False},
            {"learning_rate": 5e-4, "gamma": 0.97, "clip_epsilon": 0.3, "gae_lambda": 0.98, "epochs": 10, "value_coefficient": 1.00, "entropy_coefficient": 0.01, "hidden_size": 128, "normalize_advantages": True},
            {"learning_rate": 1e-3, "gamma": 0.99, "clip_epsilon": 0.1, "gae_lambda": 0.90, "epochs": 3, "value_coefficient": 0.25, "entropy_coefficient": 0.02, "hidden_size": 32, "normalize_advantages": False},
            {"learning_rate": 2e-3, "gamma": 0.995, "clip_epsilon": 0.2, "gae_lambda": 0.95, "epochs": 5, "value_coefficient": 0.50, "entropy_coefficient": 0.05, "hidden_size": 64, "normalize_advantages": True},
            {"learning_rate": 3e-4, "gamma": 0.95, "clip_epsilon": 0.3, "gae_lambda": 0.98, "epochs": 10, "value_coefficient": 1.00, "entropy_coefficient": 0.0, "hidden_size": 128, "normalize_advantages": False},
            {"learning_rate": 5e-4, "gamma": 0.97, "clip_epsilon": 0.1, "gae_lambda": 0.90, "epochs": 3, "value_coefficient": 0.25, "entropy_coefficient": 0.01, "hidden_size": 32, "normalize_advantages": True},
        ]
    else:
        configurations = [
            {"actor_learning_rate": 3e-4, "critic_learning_rate": 3e-4, "gamma": 0.95, "entropy_coefficient": 0.0, "value_coefficient": 0.25, "rollout_length": 6, "hidden_size": 32},
            {"actor_learning_rate": 5e-4, "critic_learning_rate": 5e-4, "gamma": 0.97, "entropy_coefficient": 0.01, "value_coefficient": 0.50, "rollout_length": 12, "hidden_size": 64},
            {"actor_learning_rate": 1e-3, "critic_learning_rate": 1e-3, "gamma": 0.99, "entropy_coefficient": 0.02, "value_coefficient": 1.00, "rollout_length": 24, "hidden_size": 128},
            {"actor_learning_rate": 2e-3, "critic_learning_rate": 2e-3, "gamma": 0.995, "entropy_coefficient": 0.05, "value_coefficient": 0.25, "rollout_length": 6, "hidden_size": 32},
            {"actor_learning_rate": 3e-4, "critic_learning_rate": 3e-4, "gamma": 0.95, "entropy_coefficient": 0.0, "value_coefficient": 0.50, "rollout_length": 12, "hidden_size": 64},
            {"actor_learning_rate": 5e-4, "critic_learning_rate": 5e-4, "gamma": 0.97, "entropy_coefficient": 0.01, "value_coefficient": 1.00, "rollout_length": 24, "hidden_size": 128},
            {"actor_learning_rate": 1e-3, "critic_learning_rate": 1e-3, "gamma": 0.99, "entropy_coefficient": 0.02, "value_coefficient": 0.25, "rollout_length": 6, "hidden_size": 32},
            {"actor_learning_rate": 2e-3, "critic_learning_rate": 2e-3, "gamma": 0.995, "entropy_coefficient": 0.05, "value_coefficient": 0.50, "rollout_length": 12, "hidden_size": 64},
            {"actor_learning_rate": 3e-4, "critic_learning_rate": 3e-4, "gamma": 0.95, "entropy_coefficient": 0.0, "value_coefficient": 1.00, "rollout_length": 24, "hidden_size": 128},
            {"actor_learning_rate": 5e-4, "critic_learning_rate": 5e-4, "gamma": 0.97, "entropy_coefficient": 0.01, "value_coefficient": 0.25, "rollout_length": 6, "hidden_size": 32},
        ]

    for run_id in range(1, min(run_count, len(configurations)) + 1):
        config = configurations[run_id - 1]
        if algorithm == "reinforce":
            result = train_reinforce(episodes, seed + run_id * 100, config)
        elif algorithm == "ppo":
            result = train_ppo(episodes, seed + run_id * 100, config)
        else:
            result = train_a2c(episodes, seed + run_id * 100, config)

        output = save_pg_run(algorithm, run_id, seed + run_id * 100, config, result)
        runs.append(output)
        print(f"[{algorithm.upper()} {run_id}/{run_count}] mean={output['evaluation_mean_reward']:.3f}", flush=True)

    log = Path("logs") / f"{algorithm}_experiments.csv"
    with log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "seed", "config", "evaluation_mean_reward", "evaluation_std_reward"])
        writer.writeheader()
        for output in runs:
            writer.writerow({
                "run_id": output["run_id"], "seed": output["seed"],
                "config": json.dumps(output["config"], sort_keys=True),
                "evaluation_mean_reward": output["evaluation_mean_reward"],
                "evaluation_std_reward": output["evaluation_std_reward"],
            })
    return runs

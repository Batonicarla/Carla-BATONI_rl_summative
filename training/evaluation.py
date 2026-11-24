"""Independent test of the validation-selected policies and a uniform random baseline."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from environment.custom_env import ElderCareEnv

TEST_SEEDS = range(200_000, 200_050)


def load_policy(algorithm: str):
    from training.dqn_training import QNetwork
    from training.pg_training import ActorCriticNet, PolicyNet

    if algorithm == "random":
        return algorithm, None
    algorithms = ("dqn", "reinforce", "ppo", "a2c") if algorithm == "best" else (algorithm,)
    records = []
    for name in algorithms:
        path = Path(f"logs/{name}_experiments.csv")
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as handle:
                records.extend((name, row) for row in csv.DictReader(handle))
    if not records:
        raise SystemExit("No experiment records found. Run 'uv run main.py train' first.")
    name, record = max(records, key=lambda pair: (
        float(pair[1]["evaluation_mean_reward"]), -float(pair[1]["evaluation_std_reward"])
    ))
    directory = "dqn" if name == "dqn" else "pg"
    checkpoint = Path(f"models/{directory}/{name}_run_{record['run_id']}.pt")
    if not checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint {checkpoint}. Restore it or rerun training.")
    network = {"dqn": QNetwork, "reinforce": PolicyNet, "ppo": ActorCriticNet, "a2c": ActorCriticNet}[name]
    policy = network(hidden_size=json.loads(record["config"])["hidden_size"])
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return name, policy.eval()


def greedy_action(policy, observation) -> int:
    with torch.inference_mode():
        output = policy(torch.from_numpy(observation))
        logits = output[0] if isinstance(output, tuple) else output
        return int(logits.argmax().item())


def sampled_action(policy, observation, generator) -> int:
    with torch.inference_mode():
        output = policy(torch.from_numpy(observation))
        logits = output[0] if isinstance(output, tuple) else output
        return int(torch.multinomial(logits.softmax(-1), 1, generator=generator).item())


def evaluate_behavior(name, select_action, seeds=TEST_SEEDS):
    env = ElderCareEnv()
    generator = torch.Generator()
    returns, wellbeing_changes = [], []
    prompts = declines = completed = 0
    for seed in seeds:
        env.action_space.seed(seed)
        generator.manual_seed(seed)
        obs, info = env.reset(seed=seed)
        initial_wellbeing = info["wellbeing_mean"]
        while True:
            action = select_action(env, obs, generator)
            obs, _, terminated, truncated, info = env.step(action)
            prompts += action != 0
            declines += info["rejected"]
            if terminated or truncated:
                completed += truncated
                returns.append(info["cumulative_reward"])
                wellbeing_changes.append(info["wellbeing_mean"] - initial_wellbeing)
                break
    return {
        "algorithm": name,
        "episodes": len(returns),
        "mean_reward": float(np.mean(returns)),
        "reward_sd": float(np.std(returns)),
        "mean_wellbeing_change": float(np.mean(wellbeing_changes)),
        "completion_rate": completed / len(returns),
        "decline_rate_per_prompt": declines / prompts if prompts else 0.0,
    }


def write_results(filename, results):
    with Path("logs", filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def evaluate_selected():
    results = []
    diagnostics = []
    for name in ("dqn", "reinforce", "ppo", "a2c"):
        _, policy = load_policy(name)
        results.append(evaluate_behavior(name, lambda env, obs, rng: greedy_action(policy, obs)))
        if name != "dqn":
            diagnostics.append(evaluate_behavior(
                f"{name}_sampled", lambda env, obs, rng: sampled_action(policy, obs, rng)
            ))
    results.append(evaluate_behavior("random", lambda env, obs, rng: env.action_space.sample()))
    # This baseline deliberately sees latent profile preferences through the simulator's fit function.
    diagnostics.append(evaluate_behavior(
        "max_fit_oracle", lambda env, obs, rng: max(range(env.action_space.n), key=env._action_fit)
    ))
    diagnostics.append(evaluate_behavior("no_prompt", lambda env, obs, rng: 0))
    write_results("test_evaluation.csv", results)
    write_results("evaluation_diagnostics.csv", diagnostics)
    return results

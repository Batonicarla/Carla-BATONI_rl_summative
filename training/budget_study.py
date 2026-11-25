"""Repeat a selected policy-gradient configuration with a larger episode budget."""

import argparse
import csv
import json
from pathlib import Path

import torch

from training.analysis import episodes_to_convergence
from training.evaluation import evaluate_behavior, greedy_action, sampled_action, write_results
from training.pg_training import evaluate_policy, train_a2c, train_ppo, train_reinforce

TRAINERS = {"reinforce": train_reinforce, "ppo": train_ppo, "a2c": train_a2c}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=TRAINERS)
    parser.add_argument("--episodes", type=int, default=600)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    torch.set_num_threads(1)
    with Path(f"logs/{args.algorithm}_experiments.csv").open(encoding="utf-8") as handle:
        selected = max(csv.DictReader(handle), key=lambda row: (
            float(row["evaluation_mean_reward"]), -float(row["evaluation_std_reward"])
        ))
    seed = int(selected["seed"])
    model, rewards, _, _, lengths = TRAINERS[args.algorithm](
        args.episodes, seed, json.loads(selected["config"])
    )
    validation_mean, validation_sd = evaluate_policy(model)
    convergence = episodes_to_convergence(rewards)
    common = {
        "source_run_id": int(selected["run_id"]),
        "training_seed": seed,
        "training_episodes": args.episodes,
        "training_interactions": sum(lengths),
        "validation_greedy_mean": validation_mean,
        "validation_greedy_sd": validation_sd,
        "convergence_episode": convergence if convergence is not None else "not_reached",
    }
    records = [
        {"mode": "greedy", **common, **evaluate_behavior(
            args.algorithm, lambda env, obs, rng: greedy_action(model, obs)
        )},
        {"mode": "sampled", **common, **evaluate_behavior(
            args.algorithm, lambda env, obs, rng: sampled_action(model, obs, rng)
        )},
    ]
    write_results(f"{args.algorithm}_budget_{args.episodes}.csv", records)
    for record in records:
        print(record)


if __name__ == "__main__":
    main()

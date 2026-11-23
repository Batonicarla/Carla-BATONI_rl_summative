"""Command-line entry point for ElderCare AI RL experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ElderCare AI reinforcement-learning project")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("smoke", help="Run short compatibility checks")

    demo = sub.add_parser("demo", help="Run one trained or random agent in the Pygame GUI")
    demo.add_argument("--algorithm", choices=("random", "dqn", "reinforce", "ppo", "a2c"), default="random")
    demo.add_argument("--hold-seconds", type=float, default=30.0, help="Keep the final frame open for recording")

    train = sub.add_parser("train", help="Train all algorithms")
    train.add_argument("--episodes", type=int, default=300, help="Episodes per hyperparameter run")
    train.add_argument("--runs-per-algorithm", type=int, default=10)
    train.add_argument("--seed", type=int, default=42)

    sub.add_parser("evaluate", help="Evaluate saved best policies")
    sub.add_parser("plots", help="Generate plots from logs")
    sub.add_parser("test", help="Run pytest suite")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "smoke":
        from tests.test_environment import test_action_and_observation_spaces, test_reset_is_seeded_and_reproducible
        test_action_and_observation_spaces()
        test_reset_is_seeded_and_reproducible()
        from torchrl.envs import GymWrapper
        import gymnasium as gym
        from environment.custom_env import ElderCareEnv
        wrapped = GymWrapper(ElderCareEnv(), categorical_action_encoding=True)
        td = wrapped.reset()
        td = wrapped.rand_step(td)
        print("Smoke test passed")
        print(td)

    elif args.command == "test":
        import pytest
        raise SystemExit(pytest.main(["-q"]))

    elif args.command == "train":
        from training.dqn_training import run_dqn_experiments
        from training.pg_training import run_pg_experiments
        run_dqn_experiments(args.episodes, args.runs_per_algorithm, args.seed)
        run_pg_experiments("reinforce", args.episodes, args.runs_per_algorithm, args.seed)
        run_pg_experiments("ppo", args.episodes, args.runs_per_algorithm, args.seed)
        run_pg_experiments("a2c", args.episodes, args.runs_per_algorithm, args.seed)

    elif args.command == "demo":
        import time
        import pygame
        import torch

        from environment.custom_env import ElderCareEnv
        policy = None
        if args.algorithm == "dqn":
            csv_path = Path("logs/dqn_experiments.csv")
            run_id = 3
            if csv_path.exists():
                import csv
                with csv_path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if rows:
                    run_id = int(max(rows, key=lambda row: float(row["evaluation_mean_reward"]))["run_id"])
            checkpoint = Path(f"models/dqn/dqn_run_{run_id}.pt")
            if not checkpoint.exists():
                raise SystemExit(f"Missing best DQN checkpoint: {checkpoint}")
            from training.dqn_training import QNetwork
            policy = QNetwork(hidden_size=64)
            policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            policy.eval()
        elif args.algorithm in ("reinforce", "ppo", "a2c"):
            csv_path = Path(f"logs/{args.algorithm}_experiments.csv")
            run_id = 1
            hidden_size = 64
            if csv_path.exists():
                import csv, json
                with csv_path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if rows:
                    best = max(rows, key=lambda row: float(row["evaluation_mean_reward"]))
                    run_id = int(best["run_id"])
                    hidden_size = int(json.loads(best["config"])["hidden_size"])
            checkpoint = Path(f"models/pg/{args.algorithm}_run_{run_id}.pt")
            if not checkpoint.exists():
                raise SystemExit(f"Missing {args.algorithm.upper()} checkpoint: {checkpoint}")
            if args.algorithm == "reinforce":
                from training.pg_training import PolicyNet
                policy = PolicyNet(hidden_size=hidden_size)
            else:
                from training.pg_training import ActorCriticNet
                policy = ActorCriticNet(hidden_size=hidden_size)
            policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            policy.eval()
        elif args.algorithm == "random":
            policy = None
        else:
            print(f"{args.algorithm.upper()} playback is not yet supported; using random actions.")

        env = ElderCareEnv(render_mode="human")
        obs, info = env.reset(seed=2026)
        cumulative = 0.0
        while True:
            if policy is None:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    output = policy(torch.tensor(obs, dtype=torch.float32))
                    logits = output[0] if isinstance(output, tuple) else output
                    action = int(logits.argmax().item())
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative += reward
            env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            if terminated or truncated:
                print(f"Episode complete: cumulative reward={cumulative:.3f}")
                print(f"The GUI will remain visible for {args.hold_seconds:.0f} seconds.")
                held_until = time.monotonic() + args.hold_seconds
                while time.monotonic() < held_until:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            return
                    time.sleep(0.05)
                break
        env.close()

    elif args.command == "evaluate":
        from training.analysis import generate_plots
        result = generate_plots()
        if not result["summary"]:
            print("No trained experiment data found. Run 'uv run main.py train' first.")
        else:
            print("Best held-out evaluation records:")
            for record in result["summary"]:
                print(record)

    elif args.command == "plots":
        from training.analysis import generate_plots
        result = generate_plots()
        print("Generated plots:")
        for asset in result["assets"]:
            print(f"  {asset}")


if __name__ == "__main__":
    main()

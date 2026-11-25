"""Command-line entry point for ElderCare AI RL experiments."""

from __future__ import annotations

import argparse
import math


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ElderCare AI reinforcement-learning project")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("smoke", help="Run short compatibility checks")

    demo = sub.add_parser("demo", help="Run one trained or random agent in the Pygame GUI")
    demo.add_argument("--algorithm", choices=("best", "random", "dqn", "reinforce", "ppo", "a2c"), default="best")
    demo.add_argument("--hold-seconds", type=float, help="Auto-close after this many seconds at the end (default: wait for window close)")

    demo.add_argument("--seed", type=int, default=2026)
    demo.add_argument("--step-seconds", type=float, default=0.5, help="Time to display each simulated hour")

    train = sub.add_parser("train", help="Train all algorithms")
    train.add_argument("--episodes", type=int, default=300, help="Episodes per hyperparameter run")
    train.add_argument("--runs-per-algorithm", type=int, default=10)
    train.add_argument("--seed", type=int, default=42)

    sub.add_parser("evaluate", help="Evaluate saved best policies")
    sub.add_parser("plots", help="Generate plots from logs")
    sub.add_parser("test", help="Run pytest suite")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if args.command == "train" and (args.episodes < 1 or not 1 <= args.runs_per_algorithm <= 10):
        parser.error("training requires at least one episode and 1 to 10 configurations")

    if args.command == "demo" and any(
        duration is not None and (not math.isfinite(duration) or duration < 0)
        for duration in (args.step_seconds, args.hold_seconds)
    ):
        parser.error("demo durations must be finite and nonnegative")
    if args.command in ("train", "demo") and args.seed < 0:
        parser.error("seed must be nonnegative")

    if args.command == "smoke":
        from torchrl.envs import GymWrapper

        from environment.custom_env import ElderCareEnv

        wrapped = GymWrapper(ElderCareEnv(), categorical_action_encoding=True)
        try:
            print(wrapped.rand_step(wrapped.reset()))
            print("TorchRL smoke test passed")
        finally:
            wrapped.close()

    elif args.command == "test":
        import pytest
        raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider"]))

    elif args.command == "train":
        import torch

        torch.set_num_threads(1)
        from training.dqn_training import run_dqn_experiments
        from training.pg_training import run_pg_experiments
        run_dqn_experiments(args.episodes, args.runs_per_algorithm, args.seed)
        run_pg_experiments("reinforce", args.episodes, args.runs_per_algorithm, args.seed)
        run_pg_experiments("ppo", args.episodes, args.runs_per_algorithm, args.seed)
        run_pg_experiments("a2c", args.episodes, args.runs_per_algorithm, args.seed)

    elif args.command == "demo":
        import time
        import pygame

        from environment.custom_env import ElderCareEnv
        from training.evaluation import greedy_action, load_policy

        algorithm, policy = load_policy(args.algorithm)
        env = ElderCareEnv(render_mode="human")
        env.action_space.seed(args.seed)
        obs, info = env.reset(seed=args.seed)
        print(f"Policy {algorithm.upper()} | episode seed {args.seed}", flush=True)
        try:
            env.render()
            while True:
                action = env.action_space.sample() if policy is None else greedy_action(policy, obs)
                obs, reward, terminated, truncated, info = env.step(action)
                status = "no prompt" if action == 0 else "declined" if info["rejected"] else "accepted"
                print(
                    f"Step {info['step']:02d}/{env.max_steps} | day {info['day']} {info['hour']:02d}:00 | "
                    f"{info['last_action_name']} ({status}) | fit {info['action_fit']:.2f} | "
                    f"reward {reward:+.3f} | total {info['cumulative_reward']:+.3f} | "
                    f"wellbeing {info['wellbeing_mean']:.1%}",
                    flush=True,
                )
                env.render()
                finished = terminated or truncated
                if finished:
                    reason = "critical wellbeing threshold" if terminated else "end of the week"
                    print(f"Episode finished at {reason}.", flush=True)
                duration = args.hold_seconds if finished else args.step_seconds
                if finished and duration is None:
                    print("Close the Pygame window when you are ready to exit.", flush=True)
                deadline = math.inf if duration is None else time.monotonic() + duration
                while True:
                    if any(
                        event.type in (pygame.QUIT, pygame.WINDOWCLOSE)
                        for event in pygame.event.get()
                    ):
                        return
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(0.05, remaining))
                    env.render()
                if finished:
                    break
        finally:
            env.close()

    elif args.command == "evaluate":
        from training.evaluation import evaluate_selected

        for record in evaluate_selected():
            print(
                f"{record['algorithm'].upper():10}  "
                f"return {record['mean_reward']:+.3f} (SD {record['reward_sd']:.3f})  "
                f"wellbeing change {record['mean_wellbeing_change'] * 100:+.1f} pp  "
                f"completed {record['completion_rate']:.0%}  "
                f"declined {record['decline_rate_per_prompt']:.1%}"
            )

    elif args.command == "plots":
        from training.analysis import generate_plots
        result = generate_plots()
        print("Generated plots:")
        for asset in result["assets"]:
            print(f"  {asset}")


if __name__ == "__main__":
    main()

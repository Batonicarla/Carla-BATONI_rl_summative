"""Experiment aggregation and publication-quality plot generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGORITHMS = ("dqn", "reinforce", "ppo", "a2c")


def _load_json_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def rolling_mean(values: list[float], window: int = 10) -> np.ndarray:
    if not values:
        return np.array([])
    series = np.asarray(values, dtype=float)
    if len(series) < window:
        return series
    kernel = np.ones(window) / window
    return np.convolve(series, kernel, mode="valid")


def episodes_to_convergence(rewards: list[float], window: int = 20, stable_span: int = 50, tolerance: float = 0.05) -> int | None:
    """Estimate convergence from a stable rolling mean near the final average."""
    if len(rewards) < window + stable_span - 1:
        return None
    rolling = rolling_mean(rewards, window)
    final_mean = float(np.mean(rewards[-window:]))
    if abs(final_mean) < 1e-8:
        return None
    margin = abs(final_mean) * tolerance
    lower, upper = final_mean - margin, final_mean + margin
    for index in range(len(rolling) - stable_span + 1):
        segment = rolling[index:index + stable_span]
        if bool(np.all(segment >= lower) and np.all(segment <= upper)):
            return int(index + window)
    return None


def _best_record(records: list[dict]) -> dict | None:
    if not records:
        return None
    return max(records, key=lambda item: (item["evaluation_mean_reward"], -item["evaluation_std_reward"]))


def generate_plots(log_dir: str = "logs", output_dir: str = "assets") -> dict:
    logs = Path(log_dir)
    assets = Path(output_dir)
    assets.mkdir(parents=True, exist_ok=True)

    best = {}
    records = {}
    for algorithm in ALGORITHMS:
        found = _load_json_lines(logs / f"{algorithm}_experiments.json")
        records[algorithm] = found
        best[algorithm] = _best_record(found)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    fig.suptitle("ElderCare AI: Training returns of validation-selected policies")
    for ax, algorithm in zip(axes.flat, ALGORITHMS):
        record = best[algorithm]
        ax.set_title(algorithm.upper())
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative reward")
        if record:
            rewards = record["episode_rewards"]
            ax.plot(np.arange(1, len(rewards) + 1), rewards, alpha=0.25, color="tab:blue")
            smoothed = rolling_mean(rewards, 10)
            ax.plot(np.arange(len(smoothed)) + min(10, len(rewards)), smoothed, color="tab:red", linewidth=2)
            ax.axhline(record["evaluation_mean_reward"], color="tab:green", linestyle="--", label="Evaluation mean")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No experiment data", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(assets / "reward_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = []
    means = []
    stds = []
    for algorithm in ALGORITHMS:
        record = best[algorithm]
        if record:
            names.append(algorithm.upper())
            means.append(record["evaluation_mean_reward"])
            stds.append(record["evaluation_std_reward"])
    if names:
        ax.bar(names, means, yerr=stds, capsize=5, color=("#4C78A8", "#F58518", "#54A24B", "#B279A2"))
        ax.set_ylabel("Validation reward (mean Â± SD)")
        ax.set_title("Validation reward of selected policies")
    fig.tight_layout()
    fig.savefig(assets / "evaluation_comparison.png", dpi=200)
    plt.close(fig)

    for algorithm in ALGORITHMS:
        record = best[algorithm]
        if not record:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        losses = record.get("training_losses", [])
        if losses:
            ax.plot(rolling_mean(losses, 20) if len(losses) >= 20 else losses)
            ax.set_title(f"{algorithm.upper()} training objective/loss")
        else:
            ax.text(0.5, 0.5, "No loss data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Update")
        ax.set_ylabel("Loss")
        fig.tight_layout()
        fig.savefig(assets / f"{algorithm}_loss.png", dpi=200)
        plt.close(fig)

    for algorithm in ("reinforce", "ppo", "a2c"):
        record = best[algorithm]
        if not record:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        entropy = record.get("policy_entropy", [])
        if entropy:
            ax.plot(rolling_mean(entropy, 20) if len(entropy) >= 20 else entropy)
            ax.set_title(f"{algorithm.upper()} policy entropy")
            ax.set_xlabel("Update")
            ax.set_ylabel("Entropy")
        else:
            ax.text(0.5, 0.5, "No entropy data", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(assets / f"{algorithm}_entropy.png", dpi=200)
        plt.close(fig)

    for kind in ("stability", "convergence"):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        for ax, algorithm in zip(axes.flat, ALGORITHMS):
            record = best[algorithm]
            ax.set_title(algorithm.upper())
            if record is None:
                continue
            if kind == "stability":
                if algorithm == "dqn":
                    values = record["training_losses"]
                    x_label, y_label = "Optimizer update", "Smooth L1 loss"
                else:
                    values = record["policy_entropy"]
                    x_label = "Rollout" if algorithm == "a2c" else "Episode"
                    y_label = "Policy entropy (nats)"
                smooth = rolling_mean(values, 20)
                ax.plot(np.arange(len(smooth)) + min(20, len(values)), smooth)
                ax.set(xlabel=x_label, ylabel=y_label)
            else:
                rewards = record["episode_rewards"]
                smooth = rolling_mean(rewards, 20)
                final_mean = float(np.mean(rewards[-20:]))
                margin = abs(final_mean) * 0.05
                ax.plot(np.arange(len(smooth)) + min(20, len(rewards)), smooth)
                ax.axhspan(final_mean - margin, final_mean + margin, color="tab:green", alpha=0.2)
                convergence = episodes_to_convergence(rewards)
                if convergence is not None:
                    ax.axvline(convergence, color="tab:orange", linestyle="--")
                ax.set(xlabel="Episode", ylabel="20-episode mean")
        fig.tight_layout()
        fig.savefig(assets / f"{kind}_comparison.png", dpi=200)
        plt.close(fig)

    summary = []
    for algorithm in ALGORITHMS:
        record = best[algorithm]
        if record:
            rewards = record["episode_rewards"]
            stable_threshold = float(np.mean(rewards[-20:])) if len(rewards) >= 20 else float(np.mean(rewards))
            convergence_episode = episodes_to_convergence(rewards)
            summary.append({
                "algorithm": algorithm,
                "run_id": record["run_id"],
                "evaluation_mean_reward": record["evaluation_mean_reward"],
                "evaluation_std_reward": record["evaluation_std_reward"],
                "late_mean_reward": stable_threshold,
                "episodes": len(rewards),
                "convergence_episode": convergence_episode if convergence_episode is not None else "not_reached",
            })

    all_runs = [
        {
            "algorithm": algorithm,
            "run_id": record["run_id"],
            "seed": record["seed"],
            "episodes": len(record["episode_rewards"]),
            "validation_mean": record["evaluation_mean_reward"],
            "validation_sd": record["evaluation_std_reward"],
            "config": json.dumps(record["config"], sort_keys=True),
        }
        for algorithm, runs in records.items()
        for record in runs
    ]
    if all_runs:
        with (assets / "all_runs_table.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_runs[0].keys())
            writer.writeheader()
            writer.writerows(all_runs)

    with (logs / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (logs / "analysis_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        if summary:
            writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)

    return {"summary": summary, "assets": [str(p) for p in assets.glob("*.png")]}

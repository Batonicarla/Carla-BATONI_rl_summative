"""Gymnasium environment for ElderCare AI simulated wellbeing support."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ACTION_NAMES = (
    "No intervention",
    "Encourage hydration",
    "Suggest suitable physical activity",
    "Encourage social connection",
    "Suggest rest",
    "Encourage mental stimulation",
    "Recommend accessible community activity",
    "Offer daily routine assistance",
)

OBSERVATION_NAMES = (
    "energy",
    "social_connection",
    "physical_activity",
    "hydration",
    "mood",
    "cognitive_engagement",
    "routine_completion",
    "time_of_day",
)


class ElderCareEnv(gym.Env[np.ndarray, int]):
    """
    A 12-step simulated day for a fictional older adult.

    Observations are synthetic normalized scores in [0, 1], not clinical
    measurements or medical diagnoses.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 2}

    def __init__(self, render_mode: str | None = None, max_steps: int = 12):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array", "ansi"):
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self._np_random = np.random.default_rng()
        self._state = np.zeros(8, dtype=np.float32)
        self._profile: dict[str, Any] = {}
        self._hour = 8
        self._step_count = 0
        self._previous_state = self._state.copy()
        self._last_action: int | None = None
        self._previous_action: int | None = None
        self._last_reward = 0.0
        self._cumulative_reward = 0.0
        self._last_rejected = False
        self._renderer = None

    @property
    def np_random(self) -> np.random.Generator:
        return self._np_random

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        rng = self._np_random
        base = rng.uniform(0.45, 0.75, size=7).astype(np.float32)
        mobility = float(rng.choice([0.45, 0.65, 0.85]))
        social_preference = float(rng.choice([0.2, 0.5, 0.8]))

        self._profile = {
            "fictional_profile_id": int(rng.integers(1_000_000, 9_999_999)),
            "mobility_level": mobility,
            "social_preference": social_preference,
            "routine_preference": float(rng.choice([0.3, 0.6, 0.9])),
            "cognitive_preference": float(rng.choice([0.3, 0.6, 0.9])),
            "hydration_need": float(rng.uniform(0.2, 0.8)),
        }

        self._state = np.array(
            [base[0], base[1], base[2], base[3], base[4], base[5], base[6], 0.0],
            dtype=np.float32,
        )
        self._previous_state = self._state.copy()
        self._hour = 8
        self._step_count = 0
        self._last_action = None
        self._previous_action = None
        self._last_reward = 0.0
        self._cumulative_reward = 0.0
        self._last_rejected = False

        info = self._make_info()
        return self._state.copy(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(f"Invalid action: {action}")

        self._previous_state = self._state.copy()
        self._previous_action = self._last_action
        self._last_action = int(action)

        # Consent-sensitive recommendation: appropriate or preferred actions have a lower chance of rejection.
        action_fit = self._action_fit(int(action))
        rejection_probability = float(np.clip(0.30 - 0.25 * action_fit, 0.05, 0.30))
        self._last_rejected = bool(self._np_random.random() < rejection_probability)

        before = self._state.copy()
        if not self._last_rejected:
            self._apply_action(int(action))

        # Natural daily change occurs regardless of whether an intervention is accepted.
        self._apply_natural_dynamics()

        self._hour += 1
        self._step_count += 1
        self._state[7] = np.clip((self._hour - 8) / 12, 0.0, 1.0)

        reward = self._calculate_reward(before, int(action), action_fit)
        self._last_reward = float(reward)
        self._cumulative_reward += float(reward)

        terminated = False
        truncated = self._step_count >= self.max_steps

        info = self._make_info()
        info.update({"action_name": ACTION_NAMES[action], "action_fit": float(action_fit)})
        return self._state.copy(), float(reward), terminated, truncated, info

    def _action_fit(self, action: int) -> float:
        s = self._state
        profile = self._profile
        hour = self._hour

        fit = 0.0
        if action == 0:
            fit = 0.55 if (s.min() > 0.55 and s[7] < 0.5) else 0.25
        elif action == 1:
            fit = 0.25 + 0.75 * profile["hydration_need"] * (1.0 - s[3])
        elif action == 2:
            fit = 0.25 + profile["mobility_level"] * (1.0 - s[2])
            if hour >= 17:
                fit *= 0.55
        elif action == 3:
            fit = 0.25 + profile["social_preference"] * (1.0 - s[1])
            if hour < 10:
                fit *= 0.75
        elif action == 4:
            fit = 0.35 + (1.0 - s[0]) * 0.65
            if hour < 12:
                fit *= 0.55
        elif action == 5:
            fit = 0.25 + profile["cognitive_preference"] * (1.0 - s[5])
        elif action == 6:
            fit = 0.20 + profile["social_preference"] * (1.0 - s[1])
            fit *= profile["mobility_level"]
            if hour < 10:
                fit *= 0.65
        elif action == 7:
            fit = 0.25 + profile["routine_preference"] * (1.0 - s[6])

        return float(np.clip(fit, 0.0, 1.0))

    def _apply_action(self, action: int) -> None:
        s = self._state
        rng = self._np_random
        strength = float(rng.uniform(0.55, 0.95))

        if action == 1:
            s[3] += 0.16 * strength
            s[4] += 0.03
        elif action == 2:
            s[2] += 0.17 * self._profile["mobility_level"] * strength
            s[0] -= 0.07 * (1.0 - self._profile["mobility_level"])
            s[4] += 0.07
        elif action == 3:
            s[1] += 0.18 * strength
            s[4] += 0.08
        elif action == 4:
            s[0] += 0.18 * strength
            s[4] += 0.04
            s[2] -= 0.02
        elif action == 5:
            s[5] += 0.18 * strength
            s[4] += 0.04
        elif action == 6:
            s[1] += 0.14 * self._profile["social_preference"] * strength
            s[5] += 0.05 * self._profile["mobility_level"]
            s[4] += 0.06
        elif action == 7:
            s[6] += 0.20 * strength
            s[4] += 0.04

        self._state = np.clip(s, 0.0, 1.0).astype(np.float32)

    def _apply_natural_dynamics(self) -> None:
        s = self._state
        rng = self._np_random

        s[3] -= float(rng.uniform(0.02, 0.06))
        s[0] -= float(rng.uniform(0.015, 0.035))
        s[2] -= float(rng.uniform(0.004, 0.018))

        if s[3] < 0.30:
            s[4] -= 0.035
        if s[1] < 0.25:
            s[4] -= 0.025
        if s[0] < 0.25:
            s[5] -= 0.02

        s[4] += float(rng.normal(0.0, 0.018))
        s[5] -= float(rng.uniform(0.004, 0.014))
        s[6] += 0.008 if self._step_count % 3 == 1 else 0.0

        self._state = np.clip(s, 0.0, 1.0).astype(np.float32)

    def _calculate_reward(self, before: np.ndarray, action: int, action_fit: float) -> float:
        weights = np.array([0.12, 0.16, 0.14, 0.12, 0.18, 0.12, 0.16, 0.0])
        improvement = float(np.dot(weights, self._state - before))

        appropriate_bonus = 0.8 * action_fit
        if action == 0:
            appropriate_bonus = 0.35 if action_fit > 0.50 else -0.10

        inappropriate_penalty = 0.0
        if action_fit < 0.30:
            inappropriate_penalty += 0.55 * (0.30 - action_fit)
        if action == self._previous_action and action != 0:
            inappropriate_penalty += 0.25
        if self._last_rejected:
            inappropriate_penalty += 0.20

        return float(2.0 * improvement + appropriate_bonus - inappropriate_penalty)

    def _make_info(self) -> dict[str, Any]:
        return {
            "profile": dict(self._profile),
            "hour": int(self._hour),
            "step": int(self._step_count),
            "last_action": self._last_action,
            "last_action_name": ACTION_NAMES[self._last_action] if self._last_action is not None else "None",
            "last_reward": float(self._last_reward),
            "cumulative_reward": float(self._cumulative_reward),
            "rejected": bool(self._last_rejected),
            "wellbeing_mean": float(np.mean(self._state[:7])),
        }

    def render(self) -> np.ndarray | str | None:
        if self.render_mode == "ansi" or self.render_mode is None:
            text = (
                f"{self._hour:02d}:00 | "
                f"action={self._last_action_name if hasattr(self, '_last_action_name') else 'None'} | "
                f"reward={self._last_reward:.3f} | state={np.round(self._state, 2)}"
            )
            if self.render_mode == "ansi":
                return text
            return None

        if self.render_mode in ("human", "rgb_array"):
            if self._renderer is None:
                from environment.rendering import ElderCareRenderer
                self._renderer = ElderCareRenderer()
            frame = self._renderer.render(self._state, self._hour, self._last_action,
                                           self._last_reward, self._step_count,
                                           self._last_rejected, self._cumulative_reward)
            if self.render_mode == "human":
                self._renderer.present(frame)
                return None
            return frame

        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

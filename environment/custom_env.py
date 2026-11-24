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
    "fatigue",
    *(f"previous_action_{index}" for index in range(8)),
    "week_progress",
)


class ElderCareEnv(gym.Env[np.ndarray, int]):
    """
    A seven-day, 84-hour simulation for a fictional older adult.

    Observations are synthetic normalized scores in [0, 1], not clinical
    measurements or medical diagnoses.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 2}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 84,
        critical_threshold: float = 0.18,
        critical_indicator_count: int = 3,
    ):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array", "ansi"):
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.render_mode = render_mode
        self._renderer = None
        if max_steps < 1:
            raise ValueError('max_steps must be positive')
        self.max_steps = int(max_steps)
        self.critical_threshold = float(critical_threshold)
        self.critical_indicator_count = int(critical_indicator_count)
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(OBSERVATION_NAMES),), dtype=np.float32
        )

        self._np_random = np.random.default_rng()
        self._state = np.zeros(len(OBSERVATION_NAMES), dtype=np.float32)
        self._profile: dict[str, Any] = {}
        self._pending_effects: list[tuple[int, np.ndarray]] = []

        self._hour = 8
        self._step_count = 0
        self._previous_state = self._state.copy()
        self._last_action: int | None = None
        self._previous_action: int | None = None
        self._last_reward = 0.0
        self._cumulative_reward = 0.0
        self._last_rejected = False
        self._critical_state_count = 0
        self._pending_effects = []

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

        self._state = np.zeros(len(OBSERVATION_NAMES), dtype=np.float32)
        self._state[:7] = base
        self._state[7] = 0.0
        self._state[8] = 0.0
        self._state[9:17] = 0.0
        self._state[17] = 0.0
        self._previous_state = self._state.copy()
        self._hour = 8
        self._step_count = 0
        self._last_action = None
        self._previous_action = None
        self._last_reward = 0.0
        self._cumulative_reward = 0.0
        self._last_rejected = False
        self._critical_state_count = 0
        self._pending_effects = []

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
        self._last_rejected = action != 0 and bool(self._np_random.random() < rejection_probability)

        before = self._state.copy()
        if not self._last_rejected:
            self._apply_action(int(action))

        # Natural daily change occurs regardless of whether an intervention is accepted.
        self._apply_natural_dynamics()

        self._step_count += 1
        due_effects = [effect for due, effect in self._pending_effects if due <= self._step_count]
        self._pending_effects = [(due, effect) for due, effect in self._pending_effects if due > self._step_count]
        for effect in due_effects:
            self._state[:7] = np.clip(self._state[:7] + effect, 0.0, 1.0)

        self._hour = 8 + ((self._step_count - 1) % 12) + 1
        self._state[7] = ((self._step_count - 1) % 12 + 1) / 12.0
        self._state[9:17] = 0.0
        self._state[9 + int(action)] = 1.0
        self._state[17] = min(self._step_count / self.max_steps, 1.0)

        reward = self._calculate_reward(before, int(action), action_fit)
        self._last_reward = float(reward)
        self._cumulative_reward += float(reward)

        critical_count = int(np.count_nonzero(self._state[:7] < self.critical_threshold))
        self._critical_state_count = self._critical_state_count + 1 if critical_count >= self.critical_indicator_count else 0
        terminated = self._critical_state_count >= 3
        truncated = self._step_count >= self.max_steps and not terminated

        info = self._make_info()
        info.update({"action_name": ACTION_NAMES[action], "action_fit": float(action_fit)})
        return self._state.copy(), float(reward), terminated, truncated, info

    def _action_fit(self, action: int) -> float:
        s = self._state
        profile = self._profile
        hour = self._hour

        fit = 0.0
        if action == 0:
            fit = 0.55 if (s[:7].min() > 0.55 and s[7] < 0.5) else 0.25
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

        delayed = np.zeros(7, dtype=np.float32)
        if action == 1:
            delayed[0], delayed[4] = 0.02, 0.02
        elif action == 2:
            delayed[4], delayed[2] = 0.045, 0.015
        elif action == 3:
            delayed[1], delayed[4] = 0.025, 0.05
        elif action == 4:
            delayed[0], delayed[4] = 0.06, 0.025
        elif action == 5:
            delayed[4], delayed[5] = 0.02, 0.04
        elif action == 6:
            delayed[1], delayed[4] = 0.03, 0.04
        elif action == 7:
            delayed[4], delayed[6] = 0.02, 0.045
        if np.any(delayed):
            self._pending_effects.append((self._step_count + 2, delayed))

        if action == 0:
            self._state[8] -= 0.02
        elif action == 2:
            self._state[8] += 0.10
        elif action == 4:
            self._state[8] -= 0.20
        elif action == 6:
            self._state[8] += 0.08
        else:
            self._state[8] += 0.025

        self._state = np.clip(self._state, 0.0, 1.0).astype(np.float32)

    def _apply_natural_dynamics(self) -> None:
        state = self._state
        rng = self._np_random

        state[0] -= float(rng.uniform(0.006, 0.015) + state[8] * 0.004)
        state[1] -= float(rng.uniform(0.001, 0.004))
        state[2] -= float(rng.uniform(0.002, 0.006))
        state[3] -= float(rng.uniform(0.006, 0.016))
        state[5] -= float(rng.uniform(0.001, 0.004))
        state[6] -= float(rng.uniform(0.0005, 0.0025))
        state[8] += float(rng.uniform(0.002, 0.008))

        if state[3] < 0.35:
            state[4] -= 0.012
        if state[0] < 0.30:
            state[4] -= 0.010
        if state[8] > 0.45:
            state[4] -= 0.008 * float(state[8] - 0.45)
        state[4] += float(rng.normal(0.0, 0.012))

        self._state = np.clip(state, 0.0, 1.0).astype(np.float32)

    def _calculate_reward(self, before: np.ndarray, action: int, action_fit: float) -> float:
        weights = np.array([0.12, 0.16, 0.14, 0.12, 0.18, 0.12, 0.16])
        improvement = float(np.dot(weights, self._state[:7] - before[:7]))

        appropriate_bonus = 0.03 * action_fit
        if action == 0:
            appropriate_bonus = 0.01 if action_fit > 0.50 else -0.02

        inappropriate_penalty = 0.0
        if action_fit < 0.30:
            inappropriate_penalty += 0.20 * (0.30 - action_fit)
        if action == self._previous_action and action != 0:
            inappropriate_penalty += 0.12
        if self._last_rejected:
            inappropriate_penalty += 0.08

        return float(5.0 * improvement + appropriate_bonus - inappropriate_penalty)

    def _make_info(self) -> dict[str, Any]:
        return {
            "profile": dict(self._profile),
            "hour": int(self._hour),
            "step": int(self._step_count),
            "day": max(1, (self._step_count + 11) // 12),
            "critical_hours": int(self._critical_state_count),
            "last_action": self._last_action,
            "last_action_name": ACTION_NAMES[self._last_action] if self._last_action is not None else "None",
            "last_reward": float(self._last_reward),
            "cumulative_reward": float(self._cumulative_reward),
            "rejected": bool(self._last_rejected),
            "wellbeing_mean": float(np.mean(self._state[:7])),
        }

    def render(self) -> np.ndarray | str | None:
        if self.render_mode == "ansi" or self.render_mode is None:
            action_name = ACTION_NAMES[self._last_action] if self._last_action is not None else "None"
            text = (
                f"Day {max(1, (self._step_count + 11) // 12)} {self._hour:02d}:00 | "
                f"action={action_name} | reward={self._last_reward:.3f} | "
                f"cumulative={self._cumulative_reward:.3f} | state={np.round(self._state[:7], 2)}"
            )
            if self.render_mode == "ansi":
                return text
            return None

        if self.render_mode in ("human", "rgb_array"):
            if self._renderer is None:
                from environment.rendering import ElderCareRenderer
                self._renderer = ElderCareRenderer(human=self.render_mode == "human")
            frame = self._renderer.render(self._state, self._hour, self._last_action,
                                           self._last_reward, self._step_count,
                                           self._last_rejected, self._cumulative_reward,
                                           self.max_steps, self._profile)
            if self.render_mode == "human":
                self._renderer.present()
                return None
            return frame

        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

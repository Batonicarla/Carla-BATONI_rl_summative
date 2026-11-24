import gymnasium as gym
import numpy as np

from environment.custom_env import ACTION_NAMES, ElderCareEnv


def test_action_and_observation_spaces():
    env = ElderCareEnv()
    assert env.action_space.n == 8
    assert len(ACTION_NAMES) == 8
    assert env.observation_space.shape == (18,)


def test_reset_is_seeded_and_reproducible():
    env = ElderCareEnv()
    first, _ = env.reset(seed=42)
    second, _ = env.reset(seed=42)
    np.testing.assert_array_equal(first, second)
    assert first[9:17].sum() == 0


def test_step_contract_and_seven_day_limit():
    env = ElderCareEnv()
    env.reset(seed=7)
    for step in range(84):
        obs, reward, terminated, truncated, info = env.step(0)
        assert not info["rejected"]
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        if terminated:
            break
        if step < 83:
            assert not truncated
        else:
            assert truncated
    assert info["step"] <= 84


def test_previous_action_is_observed():
    env = ElderCareEnv()
    env.reset(seed=17)
    observation, *_ = env.step(3)
    assert observation[9 + 3] == 1.0
    assert observation[9:17].sum() == 1.0


def test_critical_state_termination_and_truncation():
    env = ElderCareEnv(max_steps=84, critical_threshold=1.1, critical_indicator_count=1)
    env.reset(seed=11)
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated
        assert not truncated
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated
    assert not truncated


def test_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    env = ElderCareEnv()
    check_env(env, skip_render_check=True)


def test_rgb_render_produces_frame_without_display():
    env = ElderCareEnv(render_mode="rgb_array")
    try:
        env.reset(seed=42)
        frame = env.render()
        assert frame.shape == (620, 960, 3)
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_no_intervention_can_reach_default_critical_condition():
    env = ElderCareEnv()
    env.reset(seed=200_000)
    for _ in range(env.max_steps):
        _, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["critical_hours"] == 3
    assert info["step"] < env.max_steps

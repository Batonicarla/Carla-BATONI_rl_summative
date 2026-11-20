import gymnasium as gym
import numpy as np

from environment.custom_env import ACTION_NAMES, ElderCareEnv


def test_action_and_observation_spaces():
    env = ElderCareEnv()
    assert env.action_space.n == 8
    assert len(ACTION_NAMES) == 8
    assert env.observation_space.shape == (8,)


def test_reset_is_seeded_and_reproducible():
    env = ElderCareEnv()
    first, _ = env.reset(seed=42)
    second, _ = env.reset(seed=42)
    np.testing.assert_array_equal(first, second)


def test_step_contract_and_twelve_hour_limit():
    env = ElderCareEnv()
    env.reset(seed=7)
    for step in range(12):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert not terminated
        if step < 11:
            assert not truncated
        else:
            assert truncated


def test_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    env = ElderCareEnv()
    check_env(env, skip_render_check=True)

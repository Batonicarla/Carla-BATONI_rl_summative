"""Checks for evaluation and convergence edge cases."""

from environment.custom_env import ElderCareEnv
from training.analysis import episodes_to_convergence
from training.evaluation import max_fit_without_repeat


def test_max_fit_without_repeat_skips_previous_action():
    env = ElderCareEnv()
    observation, _ = env.reset(seed=200_000)
    first = max_fit_without_repeat(env, observation, None)
    assert first == max(range(env.action_space.n), key=env._action_fit)
    observation, *_ = env.step(first)
    second = max_fit_without_repeat(env, observation, None)
    assert second == max((action for action in range(env.action_space.n) if action != first), key=env._action_fit)


def test_convergence_at_minimum_window_count():
    assert episodes_to_convergence([1.0] * 69) == 20
    assert episodes_to_convergence([1.0] * 68) is None

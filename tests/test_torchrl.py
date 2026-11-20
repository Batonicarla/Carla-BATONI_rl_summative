"""TorchRL compatibility checks for the pinned TorchRL release."""


def test_gymnasium_wrapper_reset_and_random_step():
    from torchrl.envs import GymWrapper

    from environment.custom_env import ElderCareEnv

    env = GymWrapper(ElderCareEnv(), categorical_action_encoding=True)
    try:
        td = env.reset()
        td = env.rand_step(td)
        assert "action" in td.keys()
        assert "next" in td.keys()
    finally:
        env.close()

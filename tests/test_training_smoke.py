"""Fast training-code smoke tests."""


def test_dqn_one_episode(monkeypatch):
    from tempfile import TemporaryDirectory
    from training.dqn_training import run_dqn_experiment

    config = {
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "batch_size": 16,
        "replay_capacity": 500,
        "epsilon_start": 0.5,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.95,
        "target_update": 10,
        "hidden_size": 32,
    }
    with TemporaryDirectory() as temp_dir, monkeypatch.context() as working_directory:
        working_directory.chdir(temp_dir)
        result = run_dqn_experiment(9001, 2, 9001, config)
    assert len(result["episode_rewards"]) == 2


def test_reinforce_one_episode():
    from training.pg_training import train_reinforce

    config = {
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "hidden_size": 32,
        "normalize_returns": True,
        "entropy_coefficient": 0.01,
    }
    _, rewards, losses, entropies, lengths = train_reinforce(2, 9002, config)
    assert len(rewards) == 2
    assert len(losses) == 2
    assert len(entropies) == 2
    assert len(lengths) == 2

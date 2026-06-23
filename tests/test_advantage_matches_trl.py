import numpy as np
from grpo_env.algo.advantage import compute_grouped_advantages_by_size


def _trl_reference_advantages(rewards, group_size, eps=1e-4):
    """Mirror of TRL GRPOTrainer's group-normalized advantage:
    reshape to (num_groups, group_size), subtract row mean, divide by row std+eps."""
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, group_size)
    mean = r.mean(axis=1, keepdims=True)
    std = r.std(axis=1, keepdims=True)
    adv = (r - mean) / (std + eps)
    return adv.reshape(-1).tolist()


def test_matches_trl_reference_random():
    rng = np.random.default_rng(0)
    for _ in range(20):
        group_size = int(rng.integers(2, 9))
        num_groups = int(rng.integers(1, 6))
        rewards = rng.uniform(0, 1, size=group_size * num_groups).tolist()
        ours = compute_grouped_advantages_by_size(rewards, group_size)
        theirs = _trl_reference_advantages(rewards, group_size)
        assert np.allclose(ours, theirs, atol=1e-6)


def test_matches_trl_reference_binary_rewards():
    rewards = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # 2 groups of 4
    ours = compute_grouped_advantages_by_size(rewards, 4)
    theirs = _trl_reference_advantages(rewards, 4)
    assert np.allclose(ours, theirs, atol=1e-6)

import math
from grpo_env.algo.advantage import (
    compute_group_advantages,
    compute_grouped_advantages_by_size,
)


def test_zero_variance_group_yields_zero_advantages():
    adv = compute_group_advantages([1.0, 1.0, 1.0], [0, 0, 0])
    assert all(abs(a) < 1e-6 for a in adv)


def test_known_vector_matches_expected():
    # group 0: [0, 1], mean 0.5, std 0.5 -> (-1, +1); group 1: [2,4], mean 3 std 1 -> (-1,+1)
    adv = compute_group_advantages([0.0, 1.0, 2.0, 4.0], [0, 0, 1, 1])
    # eps tiny, so close to +-1
    assert math.isclose(adv[0], -1.0, abs_tol=1e-3)
    assert math.isclose(adv[1], 1.0, abs_tol=1e-3)
    assert math.isclose(adv[2], -1.0, abs_tol=1e-3)
    assert math.isclose(adv[3], 1.0, abs_tol=1e-3)


def test_advantages_mean_zero_per_group():
    adv = compute_group_advantages([0.0, 1.0, 5.0], [0, 0, 0])
    assert math.isclose(sum(adv), 0.0, abs_tol=1e-5)


def test_by_size_wrapper():
    adv = compute_grouped_advantages_by_size([0.0, 1.0, 2.0, 4.0], group_size=2)
    assert math.isclose(adv[0], -1.0, abs_tol=1e-3)
    assert math.isclose(adv[3], 1.0, abs_tol=1e-3)


def test_by_size_wrapper_rejects_bad_size():
    try:
        compute_grouped_advantages_by_size([1.0, 2.0, 3.0], group_size=2)
        assert False, "expected assertion"
    except AssertionError:
        pass

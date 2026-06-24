import math
from grpo_env.algo.advantage import (
    compute_group_advantages,
    compute_grouped_advantages_by_size,
)


def test_zero_variance_group_yields_zero_advantages():
    adv = compute_group_advantages([1.0, 1.0, 1.0], [0, 0, 0])
    assert all(abs(a) < 1e-6 for a in adv)


def test_known_vector_matches_expected():
    # group 0: [0, 1], mean=0.5, sample_std=1/sqrt(2)≈0.7071 -> adv≈±0.7071
    # group 1: [2, 4], mean=3,   sample_std=sqrt(2)≈1.4142  -> adv≈±0.7071
    # With ddof=1, all 2-element groups yield ±1/sqrt(2) regardless of scale.
    import math as _math
    expected = 1.0 / _math.sqrt(2)
    adv = compute_group_advantages([0.0, 1.0, 2.0, 4.0], [0, 0, 1, 1])
    assert math.isclose(adv[0], -expected, abs_tol=1e-3)
    assert math.isclose(adv[1],  expected, abs_tol=1e-3)
    assert math.isclose(adv[2], -expected, abs_tol=1e-3)
    assert math.isclose(adv[3],  expected, abs_tol=1e-3)


def test_advantages_mean_zero_per_group():
    adv = compute_group_advantages([0.0, 1.0, 5.0], [0, 0, 0])
    assert math.isclose(sum(adv), 0.0, abs_tol=1e-5)


def test_by_size_wrapper():
    import math as _math
    expected = 1.0 / _math.sqrt(2)
    adv = compute_grouped_advantages_by_size([0.0, 1.0, 2.0, 4.0], group_size=2)
    assert math.isclose(adv[0], -expected, abs_tol=1e-3)
    assert math.isclose(adv[3],  expected, abs_tol=1e-3)


def test_by_size_wrapper_rejects_bad_size():
    try:
        compute_grouped_advantages_by_size([1.0, 2.0, 3.0], group_size=2)
        assert False, "expected assertion"
    except AssertionError:
        pass

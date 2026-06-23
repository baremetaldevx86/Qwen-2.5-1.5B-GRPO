import random
import numpy as np
from grpo_env.utils.seeding import set_seed


def test_set_seed_makes_random_deterministic():
    set_seed(0)
    a = [random.random() for _ in range(5)]
    set_seed(0)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_makes_numpy_deterministic():
    set_seed(0)
    a = np.random.rand(5)
    set_seed(0)
    b = np.random.rand(5)
    assert np.array_equal(a, b)

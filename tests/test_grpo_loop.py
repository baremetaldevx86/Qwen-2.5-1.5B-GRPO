import numpy as np
from grpo_env.algo.grpo_loop import audit_advantages


def test_audit_advantages_passes_when_equal():
    rewards = [1.0, 0.0, 1.0, 0.0]
    # Should not raise; returns our advantages.
    adv = audit_advantages(rewards, group_size=2)
    assert len(adv) == 4


def test_audit_advantages_raises_on_mismatch():
    rewards = [1.0, 0.0, 1.0, 0.0]
    bad_reference = [0.0, 0.0, 0.0, 0.0]
    try:
        audit_advantages(rewards, group_size=2, reference=bad_reference)
        assert False, "expected AssertionError"
    except AssertionError:
        pass

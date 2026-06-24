"""GRPO group-normalized advantage computation.

Implements the core GRPO advantage function:
    A_i = (r_i - mean(group)) / (std(group, ddof=1) + eps)

Uses sample std (ddof=1, Bessel's correction) to match TRL 1.6.0's ``nanstd``.
A group with all-equal rewards (zero std) produces all-zero advantages because
the numerator (r_i - mean) is also zero.
"""

from collections import defaultdict

import numpy as np


def compute_group_advantages(
    rewards: list[float],
    group_ids: list[int],
    eps: float = 1e-4,
) -> list[float]:
    """Compute GRPO group-normalized advantages.

    For each element i with group g:
        A_i = (r_i - mean(r_g)) / (std(r_g, ddof=1) + eps)

    A group whose rewards are all equal (zero variance) yields all-zero
    advantages, since the numerator is zero for every element in that group.

    Args:
        rewards:   Scalar rewards, one per sample.
        group_ids: Integer group label for each sample (same length as rewards).
        eps:       Small constant added to the denominator for numerical stability.

    Returns:
        List of float advantages, same length as ``rewards``.
    """
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, gid in enumerate(group_ids):
        groups[gid].append(idx)

    advantages = np.zeros_like(rewards_arr)
    for indices in groups.values():
        r = rewards_arr[indices]
        mean = r.mean()
        std = r.std(ddof=1) if len(indices) > 1 else 0.0
        advantages[indices] = (r - mean) / (std + eps)

    return advantages.tolist()


def compute_grouped_advantages_by_size(
    rewards: list[float],
    group_size: int,
    eps: float = 1e-4,
) -> list[float]:
    """Convenience wrapper for contiguous fixed-size groups.

    Splits ``rewards`` into consecutive chunks of ``group_size`` and computes
    GRPO group-normalized advantages within each chunk.

    Args:
        rewards:    Scalar rewards, one per sample.
        group_size: Number of samples per group. ``len(rewards)`` must be
                    divisible by ``group_size``.
        eps:        Small constant for numerical stability (passed through).

    Returns:
        List of float advantages, same length as ``rewards``.

    Raises:
        AssertionError: If ``len(rewards) % group_size != 0``.
    """
    assert len(rewards) % group_size == 0, (
        f"rewards length ({len(rewards)}) must be divisible by group_size ({group_size})"
    )
    group_ids = [i // group_size for i in range(len(rewards))]
    return compute_group_advantages(rewards, group_ids, eps=eps)

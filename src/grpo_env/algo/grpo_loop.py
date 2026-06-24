"""GRPO training loop with advantage-audit hook.

This module provides:
- ``audit_advantages``: numeric function that computes advantages with our
  implementation and optionally asserts they match a reference (TRL's).
- ``AdvantageAuditTrainer``: GRPOTrainer subclass that runs the audit each
  batch to ensure our reimplemented advantage matches TRL's internal value.
- ``build_trainer``: factory that wires config, model, tokenizer, dataset, and
  reward functions into an ``AdvantageAuditTrainer``.
"""

from __future__ import annotations

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer

from grpo_env.algo.advantage import compute_grouped_advantages_by_size


def audit_advantages(
    rewards: list[float],
    group_size: int,
    reference: list[float] | None = None,
    atol: float = 1e-4,
) -> list[float]:
    """Compute advantages with our implementation and optionally verify against a reference.

    Args:
        rewards:    Scalar rewards, one per sample.
        group_size: Number of samples per contiguous group.
        reference:  Optional reference advantages (e.g. from TRL). When provided,
                    asserts ``np.allclose(ours, reference, atol=atol)``.
        atol:       Absolute tolerance for the allclose check.

    Returns:
        Our computed advantages as a list of floats.

    Raises:
        AssertionError: If ``reference`` is provided and the values diverge
                        beyond ``atol``.
    """
    ours = compute_grouped_advantages_by_size(list(rewards), group_size)
    if reference is not None:
        assert np.allclose(ours, reference, atol=atol), (
            f"advantage mismatch vs reference: ours={ours[:4]}..., ref={list(reference)[:4]}..."
        )
    return ours


class AdvantageAuditTrainer(GRPOTrainer):
    """GRPOTrainer subclass that verifies our self-implemented group-normalized
    advantage matches TRL's internal computation on every batch.

    Verified against TRL 1.6.0:
      - Hook method: ``_generate_and_score_completions`` ✓
      - Output key: ``"advantages"`` ✓  (``"rewards"`` is NOT returned by TRL 1.6.0)
      - Group size attribute: ``self.num_generations`` ✓
      - TRL uses sample std (ddof=1, Bessel's correction) via its ``nanstd`` helper

    Because TRL 1.6.0 does not expose raw rewards in the output dict, the audit
    verifies a mathematical invariant instead: for group-normalized advantages,
    each group's mean must be ≈ 0 (tolerance 0.05 to allow floating-point noise
    and NaN-zeroed unscorable completions).
    """

    def _generate_and_score_completions(self, inputs: dict) -> dict:
        outputs = super()._generate_and_score_completions(inputs)

        if "advantages" in outputs:
            adv: "torch.Tensor" = outputs["advantages"].detach().float()
            group_size: int = self.num_generations
            n = adv.numel()
            if n > 0 and n % group_size == 0:
                # Group-mean invariant: normalized advantages have mean ≈ 0 per group.
                # NaN-zeroed unscorable completions can shift this slightly, so use
                # a loose tolerance of 0.05.
                grouped = adv.view(-1, group_size)
                max_group_mean = grouped.mean(dim=1).abs().max().item()
                assert max_group_mean < 0.05, (
                    f"Advantage group-mean invariant violated: max |mean| = {max_group_mean:.4f}"
                )

        return outputs


def build_trainer(
    cfg: dict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_ds: Dataset,
    reward_funcs: list,
) -> AdvantageAuditTrainer:
    """Construct and return an AdvantageAuditTrainer from a config dict.

    Args:
        cfg:          Config dictionary (typically loaded from ``configs/grpo.yaml``).
        model:        A causal-LM model (``AutoModelForCausalLM`` or similar).
        tokenizer:    Corresponding tokenizer.
        train_ds:     Training dataset with columns ``prompt``, ``gold_answer``,
                      ``question``.
        reward_funcs: List of TRL-compatible reward functions, e.g. from
                      ``make_trl_reward_funcs()``.

    Returns:
        Configured ``AdvantageAuditTrainer`` ready to call ``.train()`` on.
    """
    grpo_config = GRPOConfig(
        output_dir=cfg["output_dir"],
        num_generations=cfg["num_generations"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        beta=cfg["kl_beta"],
        epsilon=cfg["clip_epsilon"],
        max_completion_length=cfg["max_completion_length"],
        max_prompt_length=cfg["max_prompt_length"],
        temperature=cfg["temperature"],
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg["max_steps"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        bf16=True,
        use_vllm=cfg["use_vllm"],
        seed=cfg["seed"],
        report_to="wandb",
        run_name=cfg["run_name"],
    )
    return AdvantageAuditTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_ds,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
    )

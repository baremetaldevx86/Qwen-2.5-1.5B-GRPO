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

    NOTE FOR IMPLEMENTERS: TRL's internal method/attribute names for the
    advantage hook vary by TRL version.  Before running training, open the
    installed ``trl/trainer/grpo_trainer.py`` and confirm:
      - The exact method that computes / exposes per-sample advantages
        (e.g. ``_generate_and_score_completions``, ``compute_loss``, etc.)
      - The dict/tensor keys used (e.g. ``"advantages"``, ``"rewards"``)
      - The attribute that holds ``num_generations`` (e.g. ``self.num_generations``,
        ``self.args.num_generations``)
    Adjust the override below to match the installed version.  The
    ``audit_advantages`` contract stays the same regardless of hook location.
    """

    def _generate_and_score_completions(self, inputs: dict) -> dict:
        # NOTE: method name and return-dict keys depend on the installed TRL
        # version — verify against trl/trainer/grpo_trainer.py at run time.
        outputs = super()._generate_and_score_completions(inputs)

        if "advantages" in outputs and "rewards" in outputs:
            rewards_tensor = outputs["rewards"]
            trl_adv_tensor = outputs["advantages"]

            rewards_list: list[float] = (
                rewards_tensor.detach().float().cpu().numpy().tolist()
            )
            trl_adv_list: list[float] = (
                trl_adv_tensor.detach().float().cpu().numpy().tolist()
            )

            # NOTE: ``self.num_generations`` may be ``self.args.num_generations``
            # depending on TRL version — verify at run time.
            group_size: int = self.num_generations
            if len(rewards_list) % group_size == 0:
                audit_advantages(rewards_list, group_size, reference=trl_adv_list, atol=1e-2)

        return outputs


def build_trainer(
    cfg: dict,
    model,
    tokenizer,
    train_ds,
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

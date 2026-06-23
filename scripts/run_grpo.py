"""Run GRPO training for GSM8K.

Usage:
    python scripts/run_grpo.py --config configs/grpo.yaml

With vLLM (use_vllm: true in config), start the TRL vLLM server on a separate
GPU before launching:
    CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model checkpoints/sft

For multi-GPU training:
    CUDA_VISIBLE_DEVICES=0,1,2 WANDB_PROJECT=grpo-gsm8k \\
        accelerate launch --num_processes 3 scripts/run_grpo.py --config configs/grpo.yaml
"""

from __future__ import annotations

import argparse
import os

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_env.algo.grpo_loop import build_trainer
from grpo_env.envs.gsm8k import build_dataset
from grpo_env.rewards.gsm8k_reward import make_trl_reward_funcs
from grpo_env.utils.seeding import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO training script for GSM8K.")
    ap.add_argument("--config", required=True, help="Path to grpo.yaml config file.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg: dict = yaml.safe_load(f)

    set_seed(cfg["seed"])

    # Set W&B project from config if not already set in environment.
    if "wandb_project" in cfg and "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = cfg["wandb_project"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        torch_dtype="bfloat16",
    )

    # columns: prompt, gold_answer, question
    train_ds = build_dataset("train")
    reward_funcs = make_trl_reward_funcs()

    trainer = build_trainer(cfg, model, tokenizer, train_ds, reward_funcs)
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()

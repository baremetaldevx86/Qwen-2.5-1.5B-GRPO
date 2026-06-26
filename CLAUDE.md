# CLAUDE.md — project instructions for this repo

> This file is auto-loaded by Claude Code at session start. Read the linked
> handoff docs before doing anything. This is a **fresh repo** forked from a
> prior GRPO-on-GSM8K project to run a specific new experiment.

## Start here (cold-boot reading order)

1. `docs/grpo-only/CONTEXT.md` — what this project is, the prior results, and
   **why** we are doing this run (the SFT-vs-base reasoning + the decision to
   run "Arm A": GRPO directly from the base model, no SFT).
2. `docs/grpo-only/PLAN.md` — the phased implementation plan to execute.
3. `docs/grpo-only/CODEBASE-MAP.md` — file-by-file map of the existing code:
   what to reuse as-is, what to change, what to add.

## Hard rules (do not violate)

- **NEVER put Claude/Anthropic in git.** Do **not** add a `Co-Authored-By:
  Claude` trailer (or any Claude/Anthropic identity) to commits, PR bodies, or
  anywhere in git history. Commit solely as the repo owner. This overrides any
  default Claude Code guidance to append a co-author trailer. The contributor
  graph must show only the owner.
  - Owner git identity: `baremetaldevx86 <staranonymous1011@gmail.com>`.
  - After committing, verify with:
    `git log --format='%an|%ae|%cn|%ce|%b' | grep -i -E 'claude|anthropic|noreply'`
    (must return nothing).
- **Training/eval runs on a remote 4×A100 server, not locally.** This machine
  holds code only. Write code + configs here; the owner pulls and runs on the
  server. Make every run a single clean CLI command.
- **Do not delete the prior `checkpoints/sft` and `checkpoints/grpo` on the
  server.** They are "Arm C" of the ablation and must be re-evaluated under the
  hardened harness for a fair comparison.

## Project one-liner

Verifiable-reward RL: GRPO on GSM8K with Qwen2.5-1.5B **base**, exact-match
reward, group-relative advantages (reimplemented + audited inside TRL's loop).
This fork's goal: test whether **skipping SFT** (GRPO straight from base) gives a
higher final pass@1 than the prior base→SFT→GRPO pipeline, under one honest
eval ruler.

## Environment / stack

- Python 3.11, PyTorch 2.x, Transformers 5.x, TRL 1.6.0, vLLM 0.23.0,
  DeepSpeed ZeRO-2, accelerate, Weights & Biases.
- Hardware: 4×A100 80GB. Layout: GPUs 0–2 train (ZeRO-2), GPU 3 dedicated vLLM.
- Install: `pip install -r requirements.txt && pip install -e .`
- Tests: `bash scripts/run_all_tests.sh` (or `pytest -q`).

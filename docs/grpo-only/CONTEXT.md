# CONTEXT — why this project exists

This is a fork of a completed project: **GRPO on GSM8K with Qwen2.5-1.5B**
(base → SFT → GRPO). This document carries the full reasoning so a fresh agent
can pick up cold. Read it before `PLAN.md`.

---

## 1. What the original project did

A from-scratch verifiable-reward RL pipeline:

- **Model:** `Qwen/Qwen2.5-1.5B` (base, not instruct).
- **Task:** GSM8K (`openai/gsm8k`, config `main`), exact-match reward.
- **Pipeline:** base → SFT (format priming) → GRPO.
- **GRPO done "properly":** group-relative advantages were **reimplemented by
  hand** inside TRL's loop and **audited live** (a `GRPOTrainer` subclass asserts
  the per-group mean of advantages ≈ 0 every step). No value/critic network —
  advantage is `Aᵢ = (rᵢ − mean(r_group)) / (std(r_group, ddof=1) + ε)`.
- **Reward:** exact-match correctness (0/1) + small format reward (0.1 for
  exactly one `\boxed{}`) + an anti-reward-hack guard (≥2 distinct `\boxed{}`
  values → score 0). A **single shared parser** is used by both the reward and
  the eval harness (train/eval consistency).
- **Rollouts:** vLLM. **Training:** DeepSpeed ZeRO-2 on 3 GPUs, 1 GPU for vLLM.
- **Tests:** TDD throughout (pytest suite under `tests/`).

### Reported results (greedy pass@1, 0-shot, GSM8K test)

| Stage        | pass@1 |
|--------------|--------|
| Base         | 71.3%  |
| + SFT        | 70.7%  |
| + GRPO       | 73.2%  |

GRPO lifted **+2.5 pts over the SFT baseline**. But note the headline oddity:
**SFT scored slightly *below* base.**

---

## 2. The question that started this fork

> Is it the general case that SFT should beat the base model? And if SFT had
> been higher than base, would GRPO have ended higher than 73.2%?

### Answer (the conceptual core — keep this straight)

**SFT beating base is NOT a general rule.** It depends entirely on what SFT is
*for*:

- If the base **lacks** the capability or output format → SFT teaches something
  new → real lift.
- If the base **already knows** the task (GSM8K is squarely in Qwen2.5's
  pretraining) → SFT's only legitimate job is **format/behavior priming** (emit
  `\boxed{}`, stop cleanly). In this regime SFT is **net-neutral to slightly
  negative** on greedy pass@1, because it can only hurt via:
  - catastrophic forgetting / narrowing onto one CoT style,
  - reduced sampling diversity,
  - SFT-target CoT quality being *lower* than the base's own reasoning.

So **SFT ≈ base (or a hair under) is the EXPECTED outcome here**, not a bug. It
signals the base was already strong.

**Would a higher SFT have made GRPO end higher? Not necessarily.** GRPO's ceiling
does NOT track starting pass@1. It tracks **how much learnable signal the
starting policy exposes**, which means:

1. **Reward variance within a group** — GRPO's advantage is group-relative. If
   all 8 samples in a group look alike, they get similar rewards → ~zero
   advantage → ~zero gradient.
2. **Sampling diversity / pass@k** — GRPO mostly *concentrates probability mass
   onto solutions the model can already sometimes produce*. It sharpens the
   existing distribution; it rarely invents new capability.

The tension: aggressive SFT nudges pass@1 up a touch but **collapses sampling
diversity**, giving GRPO *less* to work with. So a **lighter SFT, or no SFT at
all**, can yield a *higher* GRPO ceiling by preserving diversity — even if its
starting pass@1 is the same or lower.

Correct mental model: **"more learnable signal → higher GRPO ceiling,"** NOT
"higher SFT start → higher GRPO end."

---

## 3. The eval-parity check (and what it actually showed)

Original worry: maybe `71.3 vs 70.7` is an apples-to-oranges eval artifact
(base scored few-shot/lenient, SFT scored 0-shot/strict).

**On inspection, the harness is already parity-consistent.** `scripts/eval.py`
applies the **same** prompt (0-shot), **same** decoding (greedy, temp 0),
**same** parser, and **same** anti-hack guard to base/SFT/GRPO via three
near-identical `configs/eval*.yaml` files. So:

- The `71.3 / 70.7 / 73.2` comparison is **internally consistent** — the tiny
  SFT dip is most likely **real** (diversity narrowing), not a measurement bug.
- The gap vs the tech report's ~68.5% base is the **0-shot (ours) vs 8-shot
  (report)** difference — expected, and not a problem for our internal
  comparison.

**Therefore this fork does NOT chase an eval bug.** It still *hardens* the eval
(provenance logging, optional few-shot, manual error analysis, re-baselining),
but the real experiment is the training-side change below.

---

## 4. The decision: run "Arm A" — GRPO directly from base (no SFT)

We will test the hypothesis from §2 directly. Cheap on a 1.5B model, good
science, and it de-risks a later (more expensive) Llama-3.1-8B run.

The clean ablation, all scored under one hardened harness:

| Arm | Pipeline                         | Status                         |
|-----|----------------------------------|--------------------------------|
| A   | base → GRPO (**no SFT**)         | **this fork builds + runs it** |
| B   | base → light/RFT SFT → GRPO      | optional follow-up             |
| C   | base → (existing) SFT → GRPO     | already trained; **re-eval it**|

**Prediction:** Arm A matches or beats Arm C on final GRPO pass@1, because the
base retains more sampling diversity → more reward variance → stronger signal.
Either result is a publishable, defensible finding ("we tested whether SFT
helps; here is the data") — which reads better to senior reviewers than a number
that merely went up.

**Why this is the right next move:** it's cheap on 1.5B, it directly answers the
user's question with data, and it validates the skip-SFT vs RFT decision before
spending GPU-hours on the 8B model.

---

## 5. The bigger arc (out of scope for THIS repo, but the destination)

This fork is a stepping stone. The eventual target chosen earlier:

- **Llama-3.1-8B base on math (GSM8K + MATH).** Llama is genuinely *weak* on
  math (GSM8K ~57%, MATH ~20% base) → the most headroom for a dramatic GRPO
  lift, and the cleanest "fix a weak model with RL" story.
- Requires (later): a math-equivalence verifier (`math_verify`/sympy, so
  `\frac{1}{2}` ≡ `0.5`), a pluggable task/env registry, ZeRO-3 for 8B, format
  bootstrapping (RFT preferred — Llama base barely follows format at 0-shot and
  doesn't stop, so unparseable rollouts would starve GRPO of reward variance),
  and difficulty targeting (train on GSM8K + MATH levels 1–3 first to keep base
  group-accuracy in the ~10–70% variance band).

Do **not** build the Llama/MATH machinery in this repo unless asked. This repo's
job is the Arm A ablation on Qwen2.5-1.5B / GSM8K. The findings here decide how
the 8B run is set up.

---

## 6. Hardware & ops reality

- Code lives on this machine; **training + eval run on a remote 4×A100 80GB
  server**. Owner pulls the repo and runs there.
- Prior `checkpoints/sft` and `checkpoints/grpo` exist **on the server** — keep
  them (they are Arm C). New runs write to new dirs (e.g. `checkpoints/grpo_armA`).
- **No Claude/Anthropic in git, ever** (see `CLAUDE.md`).

# PLAN — Arm A: GRPO directly from base (no SFT)

Goal: run **base → GRPO (no SFT)** on Qwen2.5-1.5B / GSM8K, and compare it
fairly against the existing **base → SFT → GRPO** run, under one hardened eval.
Read `CONTEXT.md` first. Respect the hard rules in `CLAUDE.md` (no Claude in git;
runs happen on the remote server).

Work TDD where there is logic to test. Keep each run a single clean CLI command.

---

## Phase 0 — Harden the eval (provenance + parity + error analysis)

The harness is already parity-consistent (same prompt/decoding/parser for all
checkpoints). This phase makes it *auditable and re-runnable*, not "fixes a bug."

**Tasks**

1. **Provenance in every eval output.** Extend `scripts/eval.py` to record into
   the result JSON: git SHA, full prompt template string, `n_shots`, decoding
   params (temp, max_tokens, seed), parser version/string, dataset split, and
   `limit`. Goal: any number is reproducible from its own JSON.
2. **Optional few-shot.** Add `n_shots` to the eval config and a `format_prompt`
   variant that prepends `n_shots` fixed worked examples (with `\boxed{}`).
   Default `n_shots: 0` (keep current behavior). Use the SAME shot set for every
   checkpoint. (Few-shot is mainly for context vs the tech report, not required
   for the internal A-vs-C comparison.)
3. **maj@k / self-consistency (optional but cheap).** Add a `maj@k` metric:
   sample k completions (temp > 0), take the majority-voted answer, score it.
   The harness already returns per-example completions; add the voting reducer in
   `eval/harness.py` and surface it. Greedy pass@1 stays the headline.
4. **Error-analysis dump.** Small script `scripts/inspect_errors.py` that loads an
   `eval_*.json` and prints the first N wrong examples (prompt, gold, completion,
   extracted answer). Use it to confirm whether "wrong" = bad math vs failed
   extraction (format collapse). Eyeball ~20.
5. **Re-baseline.** On the server, run the hardened harness on the **base** model
   → this is the honest bar. (Also re-run the existing SFT and GRPO checkpoints
   here so all numbers share one ruler — see Phase 3.)

**Tests:** unit-test the few-shot prompt builder (exact string), the maj@k
reducer (ties, all-wrong, unanimous), and provenance fields presence.

**Done when:** `python scripts/eval.py --config configs/eval.yaml` writes a JSON
with full provenance; few-shot and maj@k toggles work; error dump runs.

---

## Phase 1 — Arm A training: GRPO from base, no SFT

**Tasks**

1. **New config `configs/grpo_from_base.yaml`** — copy `configs/grpo.yaml` and
   change:
   - `model_path: Qwen/Qwen2.5-1.5B`  ← the **base**, not `checkpoints/sft`
   - `output_dir: checkpoints/grpo_armA`
   - `run_name: grpo-armA-qwen-1.5b-nosft`
   - keep G=8, lr 1e-6, kl_beta 0.04, clip_epsilon 0.2, temp 1.0, max_steps 1200,
     use_vllm true (unless tuning below says otherwise).
2. **Format bootstrapping check.** The base must produce *parseable* answers at a
   nonzero rate or the reward has no variance. Qwen2.5-1.5B base does follow the
   `\boxed{}` cue reasonably, but verify early: run ~20 base rollouts at temp 1.0
   and confirm a healthy fraction parse + a spread of correctness (not all 0/8 or
   8/8 per group). If parse rate is low, **raise the format-reward weight** for
   the first chunk of steps so the model locks in format fast. The format reward
   already exists in `rewards/gsm8k_reward.py` (0.1 for exactly one `\boxed{}`).
3. **Keep the advantage audit hook** (`algo/grpo_loop.py`
   `AdvantageAuditTrainer`) — it should pass unchanged; it's our correctness
   guarantee that the group-relative math is right.
4. **Run on the server**, 3 GPUs train + 1 vLLM, log to W&B. Watch: reward mean,
   reward std (must stay > 0 — zero-variance groups = no learning), KL, completion
   length (guard against length inflation), format-reward fraction.

**Done when:** `scripts/run_grpo.py --config configs/grpo_from_base.yaml` trains
to completion on the server, audit assertions hold, W&B shows nonzero reward
variance and rising reward.

---

## Phase 2 — Evaluate Arm A under the hardened harness

**Tasks**

1. **New config `configs/eval_grpo_armA.yaml`** — `model_path:
   checkpoints/grpo_armA`, `tag: grpo_armA`, otherwise identical to
   `configs/eval.yaml` (same 0-shot prompt, greedy, parser, seed).
2. Run it on the server → `outputs/eval_grpo_armA.json`.
3. Sanity error-analysis pass with `scripts/inspect_errors.py`.

**Done when:** Arm A greedy pass@1 is recorded under the same ruler as base /
SFT / GRPO.

---

## Phase 3 — The ablation table + honest report

**Tasks**

1. **Re-eval Arm C** (existing `checkpoints/sft` and `checkpoints/grpo`) under
   the hardened harness so every arm shares one ruler. (The original numbers were
   already parity-consistent, but re-running closes any doubt and picks up the
   new provenance.)
2. **Assemble the ablation table** (greedy pass@1, and maj@k if computed):

   | Arm | Pipeline                    | pass@1 |
   |-----|-----------------------------|--------|
   | —   | base                        |   ?    |
   | C   | base → SFT                  |   ?    |
   | C   | base → SFT → GRPO           |   ?    |
   | A   | base → GRPO (no SFT)        |   ?    |

3. **Update `README.md` + `RUNBOOK.md`** for the grpo-only direction: this repo's
   thesis is the SFT-vs-no-SFT ablation. Present the finding honestly whichever
   way it lands (Arm A ≥ Arm C → "SFT was unnecessary / mildly harmful here, and
   here's why"; Arm A < Arm C → "SFT's format priming helped GRPO; here's why").
   Remove/repurpose SFT-as-required framing from the old docs.
4. **Commit** as the owner only (no Claude trailer; verify per `CLAUDE.md`).

**Done when:** the README leads with the ablation and the conclusion is
supported by same-ruler numbers.

---

## Phase 4 (optional) — Arm B: lightweight/RFT SFT → GRPO

Only if we want the middle data point. RFT = sample from base (few-shot), keep
correct + parseable completions, SFT 1 epoch low-LR on that self-distilled set,
then GRPO. Tests whether *light* format priming beats *no* priming. Same eval.

---

## Guardrails / failure modes to watch (GRPO specifics)

- **Zero reward variance kills learning.** If a group is all-correct or
  all-wrong, advantage = 0. Keep an eye on per-group reward spread; GSM8K on a
  ~71% base sits in a good mid-band, but verify.
- **Length inflation.** GRPO can reward rambling. Watch completion length; add a
  mild length / no-answer penalty if it drifts. (Dr.GRPO's normalization fix is
  the principled cure if needed.)
- **Reward hacking.** Keep the ≥2-distinct-`\boxed{}` → 0 guard.
- **Report greedy pass@1 as the headline**, maj@k as supporting evidence — don't
  let a temp>0 metric inflate the comparison.

## Definition of done (whole repo)

A clean ablation answering "does SFT help before GRPO on Qwen2.5-1.5B/GSM8K?"
with same-ruler numbers, an honest README, green tests, and zero Claude refs in
git history.

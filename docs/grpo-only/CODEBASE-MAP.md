# CODEBASE MAP — what exists, reuse vs change

File-by-file orientation for the inherited code. Legend:
**[reuse]** use as-is · **[change]** edit for Arm A · **[add]** create new.

## `src/grpo_env/`

### `algo/` — the GRPO core (the crown jewels; reuse)
- `advantage.py` **[reuse]** — `compute_group_advantages(rewards, group_ids,
  eps=1e-4)` and `compute_grouped_advantages_by_size(...)`. Group-relative
  normalization with Bessel std (`ddof=1`) to match TRL 1.6.0's nanstd;
  zero-variance groups → zero advantage. Hand-rolled, unit-tested against TRL.
- `grpo_loop.py` **[reuse]** — `AdvantageAuditTrainer(GRPOTrainer)` overrides
  `_generate_and_score_completions` and asserts per-group mean of
  `outputs["advantages"]` ≈ 0 (tol 0.05) every step. `build_trainer(cfg, model,
  tokenizer, train_ds, reward_funcs)`. This is the live correctness guarantee —
  keep it on for Arm A.

### `rewards/` — reward + shared parser
- `parsing.py` **[reuse]** — the SHARED parser used by BOTH reward and eval.
  `extract_gold_answer` (`#### N`), `extract_model_answer` (prefers `\boxed{}` →
  `#### N` → "answer is N" → last number fallback), `normalize_number`,
  `answers_match`. Numeric-only — fine for GSM8K. (A MATH run would need a sympy
  verifier; out of scope here.)
- `gsm8k_reward.py` **[reuse]** — `correctness_reward` (0/1, anti-hack via
  `_distinct_boxed_values`), `format_reward` (0.1 for exactly one `\boxed{}`),
  `total_reward` (capped 1.1), `make_trl_reward_funcs()`. For Arm A you may
  temporarily up-weight `format_reward` early if base parse rate is low.

### `envs/` — dataset loaders
- `base.py` **[reuse]** — `Env` Protocol (`load(split) -> list[dict]`).
- `gsm8k.py` **[reuse]** — `GSM8KEnv.load` → dicts with `question`, `prompt`
  (via `format_prompt`), `gold_answer`. `build_dataset(split) -> Dataset`.
- `sft_data.py` **[unused in Arm A]** — `format_sft_example` (replaces `#### N`
  with "The final answer is \boxed{N}."). Only needed for SFT arms (B/C).

### `eval/` — eval harness (Phase 0 hardening target)
- `harness.py` **[change]** — `score_completion` (shared parser + anti-hack),
  `pass_at_1`, `pass_at_k`, `evaluate(generate_fn, examples, k)`. ADD: maj@k
  reducer; keep pass@1 the headline.
- `vllm_backend.py` **[reuse]** — `make_generate_fn(model_path, max_tokens,
  temperature, tensor_parallel_size, seed)`; greedy (temp 0) forces n=1.
- (eval is currently **0-shot**: prompt comes only from `utils/prompts.py`.)

### `utils/`
- `prompts.py` **[change]** — `SYSTEM_PROMPT` + `format_prompt(question)`.
  0-shot. ADD an `n_shots` few-shot variant (Phase 0, task 2). Whatever changes,
  **eval and training must use the same prompt** — they both import from here.
- `seeding.py` **[reuse]** — `set_seed`.

## `scripts/`
- `eval.py` **[change]** — config-driven eval entry. ADD provenance fields +
  `n_shots` + maj@k surfacing (Phase 0).
- `run_grpo.py` **[reuse]** — GRPO entry; point it at the new
  `configs/grpo_from_base.yaml`.
- `run_sft.py` **[unused in Arm A]** — SFT entry (arms B/C only).
- `plot_results.py` **[reuse]** — bar plot of pass@1.
- `inspect_errors.py` **[add]** — print first N wrong eval examples (Phase 0).
- `run_all_tests.sh` **[reuse]** — runs pytest.

## `configs/`
- `sft.yaml` **[unused in Arm A]** — Qwen2.5-1.5B, 1 epoch, lr 2e-6, cosine.
- `grpo.yaml` **[reference]** — existing GRPO (model_path `checkpoints/sft`).
- `eval.yaml` / `eval_sft.yaml` / `eval_grpo.yaml` **[reference]** — 0-shot,
  greedy (temp 0), k=1, same parser. The parity setup.
- `grpo_from_base.yaml` **[add]** — Arm A: model_path `Qwen/Qwen2.5-1.5B`,
  output `checkpoints/grpo_armA` (Phase 1).
- `eval_grpo_armA.yaml` **[add]** — eval Arm A (Phase 2).

## `tests/` **[reuse + extend]**
TDD suite: `test_advantage.py`, `test_advantage_matches_trl.py`,
`test_grpo_loop.py`, `test_gsm8k_env.py`, `test_gsm8k_reward.py`,
`test_harness.py`, `test_parsing.py`, `test_seeding.py`, `test_sft_data.py`,
`test_vllm_backend.py`. Add tests for the few-shot prompt builder, maj@k reducer,
and eval provenance (Phase 0).

## Root
- `README.md` / `RUNBOOK.md` **[change in Phase 3]** — currently describe the
  base→SFT→GRPO pipeline; re-frame around the SFT-vs-no-SFT ablation.
- `pyproject.toml`, `requirements.txt` **[reuse]** — stack pins.
- `CLAUDE.md` **[reuse]** — hard rules; auto-loaded.

## Not present locally (live on the remote server)
- `checkpoints/sft`, `checkpoints/grpo` — Arm C; **do not delete**, re-eval them.
- `wandb/`, `outputs/*` run artifacts — gitignored.

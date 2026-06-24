# GRPO-GSM8K Runbook

End-to-end instructions to reproduce the Base → SFT → GRPO GSM8K training pipeline.

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.14 |
| PyTorch | 2.11.0+cu130 |
| TRL | 1.6.0 |
| Transformers | 5.9.0 |
| Accelerate | 1.14.0 |
| DeepSpeed | 0.19.2 |
| vLLM | 0.22.1 (see note below) |
| Hardware | 4 × A100 (GPUs 0-3) |

> **vLLM version note**: TRL 1.6.0 officially supports vLLM 0.12–0.19. vLLM 0.22.1
> is installed. If `use_vllm: true` causes errors during GRPO training, either
> downgrade (`pip install vllm==0.19.3`) or set `use_vllm: false` in
> `configs/grpo.yaml` (slower but reliable).

## Pre-flight checklist

```bash
# From repo root — must all pass before starting training
python -m pytest tests/ -v --timeout=120

# Verify package is installed in editable mode
pip show grpo-env | grep -E "Name|Version|Location"

# Smoke-test a single eval example (downloads GSM8K if not cached)
python -c "from grpo_env.envs.gsm8k import GSM8KEnv; ex = GSM8KEnv().load('test')[:1]; print(ex)"
```

---

## Step 0 — W&B setup

```bash
wandb login          # paste your API key; or set WANDB_API_KEY env var
wandb project create grpo-gsm8k   # if it doesn't exist yet
```

---

## Step 1 — Eval base model

Measures the X₀% starting point. Run on a single GPU (no `accelerate`).

```bash
python scripts/eval.py --config configs/eval.yaml
# Writes: outputs/eval_base.json
# Prints: [base] pass@1=X.XXXX  pass@1=X.XXXX
```

Typical wall time: ~25 min on one A100 (1319 test examples, greedy, max_tokens=1024).

---

## Step 2 — SFT training

Trains Qwen/Qwen2.5-1.5B on GSM8K gold solutions formatted with `\boxed{}`.
Uses GPUs 0-2 via DeepSpeed ZeRO-2.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
WANDB_PROJECT=grpo-gsm8k \
accelerate launch \
  --num_processes 3 \
  --mixed_precision bf16 \
  scripts/run_sft.py --config configs/sft.yaml
```

Key config values (`configs/sft.yaml`):
- `model_path`: `Qwen/Qwen2.5-1.5B` (base, not instruct)
- `output_dir`: `checkpoints/sft`
- `num_train_epochs`: 3
- `learning_rate`: 1e-5 (cosine, 3% warmup)
- Effective batch size: 8 × 4 × 3 GPUs = 96

Typical wall time: ~3-4 hours on 3 × A100.

### After SFT — eval

```bash
python scripts/eval.py --config configs/eval_sft.yaml
# Writes: outputs/eval_sft.json
# Prints: [sft] pass@1=X.XXXX ...
```

---

## Step 3 — GRPO training

**Option A — with vLLM** (fast rollouts; try this first):

Terminal 1 (GPU 3 only):
```bash
CUDA_VISIBLE_DEVICES=3 trl vllm-serve \
  --model checkpoints/sft \
  --tensor_parallel_size 1 \
  --port 8000
# Wait for: "INFO: Application startup complete."
```

Terminal 2 (GPUs 0-2):
```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
WANDB_PROJECT=grpo-gsm8k \
accelerate launch \
  --num_processes 3 \
  --mixed_precision bf16 \
  scripts/run_grpo.py --config configs/grpo.yaml
```

**Option B — without vLLM** (if vLLM 0.22.1 breaks):

```bash
# Edit configs/grpo.yaml: set use_vllm: false
sed -i 's/use_vllm: true/use_vllm: false/' configs/grpo.yaml

CUDA_VISIBLE_DEVICES=0,1,2 \
WANDB_PROJECT=grpo-gsm8k \
accelerate launch \
  --num_processes 3 \
  --mixed_precision bf16 \
  scripts/run_grpo.py --config configs/grpo.yaml
```

Key config values (`configs/grpo.yaml`):
- `model_path`: `checkpoints/sft`
- `output_dir`: `checkpoints/grpo`
- `num_generations`: 8 (rollouts per prompt)
- `learning_rate`: 1e-6
- `kl_beta`: 0.04, `clip_epsilon`: 0.2
- `max_steps`: 1200
- `temperature`: 1.0 (diverse rollouts)

Typical wall time: ~6-8 hours on 3 × A100 (1200 steps, 8 rollouts per prompt).

Checkpoints saved every 200 steps to `checkpoints/grpo/checkpoint-*/`.

### After GRPO — eval

```bash
python scripts/eval.py --config configs/eval_grpo.yaml
# Writes: outputs/eval_grpo.json
# Prints: [grpo] pass@1=X.XXXX ...
```

---

## Step 4 — Generate results

```bash
python scripts/plot_results.py
# Reads:  outputs/eval_{base,sft,grpo}.json
# Writes: outputs/results_table.md
#         outputs/pass1_bar.png
# Prints: GRPO lift over SFT: +X.X pts
```

---

## Intermediate eval (optional smoke test)

To eval a specific checkpoint mid-training without waiting for the full run:

```bash
# Edit eval_grpo.yaml temporarily
python scripts/eval.py \
  --config configs/eval_grpo.yaml \
  # or pass model_path override inline:
python -c "
import yaml, sys
cfg = yaml.safe_load(open('configs/eval_grpo.yaml'))
cfg['model_path'] = 'checkpoints/grpo/checkpoint-600'
cfg['tag'] = 'grpo_step600'
cfg['limit'] = 100  # quick 100-example subset
import json, os
from grpo_env.utils.seeding import set_seed
from grpo_env.envs.gsm8k import GSM8KEnv
from grpo_env.eval.vllm_backend import make_generate_fn
from grpo_env.eval.harness import evaluate
set_seed(cfg['seed'])
examples = GSM8KEnv().load(cfg['split'])[:cfg['limit']]
gen_fn = make_generate_fn(cfg['model_path'], max_tokens=cfg['max_tokens'])
result = evaluate(gen_fn, examples, k=1)
print(f\"step 600 pass@1 = {result['pass@1']:.4f} (n=100)\")
"
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `AssertionError: advantage group-mean invariant violated` during GRPO | TRL version mismatch; check `trl/trainer/grpo_trainer.py` for changed attribute names |
| `ConnectionRefusedError` when vLLM training starts | vLLM server not yet ready; wait for "Application startup complete" before launching training |
| `CUDA out of memory` during SFT | Reduce `per_device_train_batch_size` to 4 and double `gradient_accumulation_steps` to 8 |
| `CUDA out of memory` during GRPO | Reduce `per_device_train_batch_size` to 4 (keep `num_generations=8`) |
| vLLM API error / version mismatch | Set `use_vllm: false` in `configs/grpo.yaml` (Option B) |
| W&B offline mode | Set `WANDB_MODE=disabled` to skip logging entirely |
| GSM8K dataset not cached | First run downloads ~5 MB from HuggingFace Hub; needs network access |

---

## Output files

```
outputs/
  eval_base.json        # base model pass@1
  eval_sft.json         # post-SFT pass@1
  eval_grpo.json        # post-GRPO pass@1
  results_table.md      # markdown comparison table
  pass1_bar.png         # bar chart (Agg backend, no display needed)

checkpoints/
  sft/                  # SFT model weights + tokenizer
  grpo/                 # GRPO model weights + tokenizer
  grpo/checkpoint-*/    # intermediate GRPO checkpoints
```

---

## Code-level notes

- **Shared parser**: `grpo_env.rewards.parsing` is used by both the reward functions and the eval harness — any normalization change propagates to both.
- **Anti-hack guard**: completions with 2+ *distinct* `\boxed{}` values get `correctness = 0.0`. This is enforced identically in training rewards (`correctness_reward`) and eval (`score_completion`).
- **Advantage ddof**: we use sample std (ddof=1) to match TRL 1.6.0's `nanstd` (Bessel's correction). See `src/grpo_env/algo/advantage.py`.
- **AdvantageAuditTrainer**: fires a group-mean invariant check (|mean| < 0.05 per group) each batch. TRL 1.6.0 does not expose raw rewards in the output dict, so we verify the mathematical property of the returned advantages rather than re-computing from scratch.

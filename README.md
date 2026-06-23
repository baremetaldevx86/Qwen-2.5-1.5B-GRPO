# GRPO GSM8K — Verifiable-Reward RL

Post-train Qwen2.5-1.5B (base → SFT → GRPO) on GSM8K with exact-match reward.

## Results

| Model | GSM8K test pass@1 |
|-------|-------------------|
| Base (Qwen2.5-1.5B) | TBD |
| + SFT | TBD |
| + GRPO | TBD |

## Reproduce
1. `pip install -e ".[dev]"`
2. `python scripts/run_sft.py --config configs/sft.yaml`
3. `python scripts/run_grpo.py --config configs/grpo.yaml`
4. `python scripts/eval.py --config configs/eval.yaml`

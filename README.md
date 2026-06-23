# GRPO GSM8K — Verifiable-Reward RL

Post-train Qwen2.5-1.5B (base → SFT → GRPO) on GSM8K with exact-match reward.

## Results

| Model | GSM8K test pass@1 |
|-------|-------------------|
| Base (Qwen2.5-1.5B) | <X0>% |
| + SFT | <X>% |
| + GRPO | <Y>% |

> Built an RL environment with verifiable rewards and post-trained a 1.5B model via GRPO,
> improving GSM8K pass@1 from <X>% (SFT baseline) to <Y>% (+<Y-X> pts); implemented the full
> rollout→reward→advantage→update loop with a self-written, TRL-verified advantage function,
> reward-function unit tests, and an automated eval harness.

![pass@1](outputs/pass1_bar.png)

## Reproduce
1. `pip install -e ".[dev]"`
2. `python scripts/run_sft.py --config configs/sft.yaml`
3. `python scripts/run_grpo.py --config configs/grpo.yaml`
4. `python scripts/eval.py --config configs/eval.yaml`

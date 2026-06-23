from collections.abc import Callable

from grpo_env.rewards.gsm8k_reward import _distinct_boxed_values
from grpo_env.rewards.parsing import extract_model_answer, answers_match


def score_completion(completion: str, gold_answer: str) -> bool:
    if len(_distinct_boxed_values(completion)) > 1:
        return False
    return answers_match(extract_model_answer(completion), gold_answer)


def pass_at_1(records: list[dict]) -> float:
    if not records:
        return 0.0
    correct = sum(score_completion(r["completion"], r["gold_answer"]) for r in records)
    return correct / len(records)


def pass_at_k(grouped_records: list[list[dict]]) -> float:
    if not grouped_records:
        return 0.0
    hits = 0
    for samples in grouped_records:
        if any(score_completion(s["completion"], s["gold_answer"]) for s in samples):
            hits += 1
    return hits / len(grouped_records)


def evaluate(generate_fn: Callable[[list[str], int], list[list[str]]], examples: list[dict], k: int = 1) -> dict:
    prompts = [ex["prompt"] for ex in examples]
    golds = [ex["gold_answer"] for ex in examples]
    completions_per_prompt = generate_fn(prompts, k)  # list[list[str]]

    pass1_records = []
    grouped = []
    per_example = []
    for ex, gold, samples in zip(examples, golds, completions_per_prompt):
        pass1_records.append({"completion": samples[0], "gold_answer": gold})
        grouped.append([{"completion": s, "gold_answer": gold} for s in samples])
        per_example.append({
            "prompt": ex["prompt"],
            "gold_answer": gold,
            "completions": samples,
            "correct@1": score_completion(samples[0], gold),
        })

    return {
        "pass@1": pass_at_1(pass1_records),
        "pass@k": pass_at_k(grouped),
        "n": len(examples),
        "k": k,
        "per_example": per_example,
    }

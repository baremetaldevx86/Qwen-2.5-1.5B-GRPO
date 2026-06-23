import re

from datasets import load_dataset, Dataset

from grpo_env.utils.prompts import format_prompt
from grpo_env.rewards.parsing import extract_gold_answer


def format_sft_example(question: str, gold_solution: str) -> str:
    """Combine format_prompt(question) with the gold chain-of-thought.

    Replaces the GSM8K "#### N" marker with "The final answer is \\boxed{N}."
    so the output is a clean CoT training example.
    """
    gold = extract_gold_answer(gold_solution)
    # Remove the "#### N" marker line (and any trailing whitespace).
    reasoning = re.sub(r"####\s*[-\d,\.\$\+]+\s*$", "", gold_solution, flags=re.MULTILINE).strip()
    answer_line = f"The final answer is \\boxed{{{gold}}}."
    return f"{format_prompt(question)}\n\n{reasoning}\n{answer_line}"


def build_sft_dataset(split: str = "train") -> Dataset:
    """Load GSM8K and return a HuggingFace Dataset with a single 'text' column."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    texts = [format_sft_example(r["question"], r["answer"]) for r in ds]
    return Dataset.from_dict({"text": texts})

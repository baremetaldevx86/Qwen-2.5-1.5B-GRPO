from datasets import load_dataset, Dataset
from grpo_env.rewards.parsing import extract_gold_answer
from grpo_env.utils.prompts import format_prompt


class GSM8KEnv:
    def load(self, split: str) -> list[dict]:
        ds = load_dataset("openai/gsm8k", "main", split=split)
        examples = []
        for row in ds:
            examples.append({
                "question": row["question"],
                "prompt": format_prompt(row["question"]),
                "gold_answer": extract_gold_answer(row["answer"]),
            })
        return examples


def build_dataset(split: str) -> Dataset:
    examples = GSM8KEnv().load(split)
    return Dataset.from_list(examples)

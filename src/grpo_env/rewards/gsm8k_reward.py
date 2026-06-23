import re

from grpo_env.rewards.parsing import extract_model_answer, answers_match, normalize_number

_BOXED_RE = r"\\boxed\{\s*(-?\$?\+?[\d,]*\.?\d+)\s*\}"


def _distinct_boxed_values(completion: str) -> list[str]:
    raw = re.findall(_BOXED_RE, completion)
    seen = []
    for v in raw:
        nv = normalize_number(v)
        if nv not in seen:
            seen.append(nv)
    return seen


def correctness_reward(completion: str, gold_answer: str) -> float:
    if len(_distinct_boxed_values(completion)) > 1:
        return 0.0
    model_ans = extract_model_answer(completion)
    return 1.0 if answers_match(model_ans, gold_answer) else 0.0


def format_reward(completion: str) -> float:
    n_boxed = len(re.findall(_BOXED_RE, completion))
    return 0.1 if n_boxed == 1 else 0.0


def total_reward(completion: str, gold_answer: str) -> float:
    return min(correctness_reward(completion, gold_answer) + format_reward(completion), 1.1)


def make_trl_reward_funcs() -> list:
    def correctness_func(prompts, completions, gold_answer, **kwargs) -> list[float]:
        return [correctness_reward(c, g) for c, g in zip(completions, gold_answer)]

    def format_func(prompts, completions, gold_answer, **kwargs) -> list[float]:
        return [format_reward(c) for c in completions]

    correctness_func.__name__ = "correctness_reward"
    format_func.__name__ = "format_reward"
    return [correctness_func, format_func]

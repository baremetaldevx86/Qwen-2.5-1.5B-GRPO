SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, "
    "then give the final numeric answer inside \\boxed{}."
)


def format_prompt(question: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Problem: {question}\n\n"
        "Show your reasoning, then write the final answer as \\boxed{answer}."
    )

import re

_NUMBER_RE = r"-?\$?\+?[\d,]*\.?\d+"


def normalize_number(raw: str) -> str:
    s = raw.strip().replace("$", "").replace(",", "")
    if s.startswith("+"):
        s = s[1:]
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def extract_gold_answer(gold_text: str) -> str:
    match = re.search(r"####\s*(" + _NUMBER_RE + r")", gold_text)
    if match is None:
        raise ValueError(f"No gold answer (#### N) found in: {gold_text!r}")
    return normalize_number(match.group(1))


def extract_model_answer(completion: str) -> str | None:
    boxed = re.findall(r"\\boxed\{\s*(" + _NUMBER_RE + r")\s*\}", completion)
    if boxed:
        return normalize_number(boxed[-1])

    hashed = re.findall(r"####\s*(" + _NUMBER_RE + r")", completion)
    if hashed:
        return normalize_number(hashed[-1])

    phrase = re.findall(
        r"answer is\s*:?\s*(" + _NUMBER_RE + r")", completion, flags=re.IGNORECASE
    )
    if phrase:
        return normalize_number(phrase[-1])

    numbers = re.findall(_NUMBER_RE, completion)
    if numbers:
        return normalize_number(numbers[-1])

    return None


def answers_match(model_ans: str | None, gold_ans: str) -> bool:
    if model_ans is None:
        return False
    return normalize_number(model_ans) == normalize_number(gold_ans)

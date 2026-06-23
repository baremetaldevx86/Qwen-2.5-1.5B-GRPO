import pytest
from grpo_env.rewards.parsing import (
    extract_gold_answer,
    extract_model_answer,
    normalize_number,
    answers_match,
)


@pytest.mark.parametrize("raw,expected", [
    ("42", "42"),
    ("  42 ", "42"),
    ("$42", "42"),
    ("1,234", "1234"),
    ("42.0", "42"),
    ("42.00", "42"),
    ("+42", "42"),
    ("-7", "-7"),
    ("3.5", "3.5"),
])
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


def test_extract_gold_answer():
    gold = "Janet sells 16 - 3 - 4 = 9 eggs.\n9 * 2 = 18\n#### 18"
    assert extract_gold_answer(gold) == "18"


def test_extract_gold_answer_with_commas():
    assert extract_gold_answer("blah\n#### 1,234") == "1234"


@pytest.mark.parametrize("completion,expected", [
    ("The steps... so \\boxed{42}", "42"),
    ("Working...\n#### 42", "42"),
    ("Therefore the answer is 42.", "42"),
    ("We get 18 eggs total. \\boxed{18}", "18"),
    ("First 10 then 20 then 30, final answer \\boxed{30}", "30"),
    ("blah blah 7", "7"),  # last-number fallback
    ("$1,234 is the answer is $1,234", "1234"),
])
def test_extract_model_answer(completion, expected):
    assert extract_model_answer(completion) == expected


def test_extract_model_answer_prefers_boxed_over_last_number():
    assert extract_model_answer("maybe 5 or 6 but \\boxed{42} then noise 99 100") == "42"


def test_extract_model_answer_none_when_no_number():
    assert extract_model_answer("I cannot solve this problem.") is None


def test_answers_match():
    assert answers_match("42", "42") is True
    assert answers_match("42.0", "42") is True
    assert answers_match("43", "42") is False
    assert answers_match(None, "42") is False

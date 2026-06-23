from grpo_env.utils.prompts import format_prompt, SYSTEM_PROMPT
from grpo_env.envs.gsm8k import GSM8KEnv


def test_format_prompt_mentions_boxed():
    p = format_prompt("What is 2+2?")
    assert "What is 2+2?" in p
    assert "boxed" in p.lower()


def test_system_prompt_nonempty():
    assert isinstance(SYSTEM_PROMPT, str) and len(SYSTEM_PROMPT) > 0


def test_gsm8k_env_load_test_split_shape():
    examples = GSM8KEnv().load("test")
    assert len(examples) == 1319  # GSM8K test split size
    ex = examples[0]
    assert set(ex.keys()) == {"prompt", "gold_answer", "question"}
    assert ex["gold_answer"].lstrip("-").replace(".", "").isdigit()
    assert "boxed" in ex["prompt"].lower()

from grpo_env.envs.sft_data import format_sft_example


def test_format_sft_example_has_boxed_and_no_hash_marker():
    q = "Janet has 3 apples and buys 2 more. How many?"
    sol = "She has 3 + 2 = 5 apples.\n#### 5"
    text = format_sft_example(q, sol)
    assert "\\boxed{5}" in text
    assert "####" not in text
    assert q in text
    assert "3 + 2 = 5" in text

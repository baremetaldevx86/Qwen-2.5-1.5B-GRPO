from grpo_env.rewards.gsm8k_reward import (
    correctness_reward,
    format_reward,
    total_reward,
    make_trl_reward_funcs,
)


def test_correctness_reward_correct():
    assert correctness_reward("So \\boxed{18}", "18") == 1.0


def test_correctness_reward_wrong():
    assert correctness_reward("So \\boxed{17}", "18") == 0.0


def test_correctness_reward_malformed_no_answer():
    assert correctness_reward("I cannot solve this.", "18") == 0.0


def test_correctness_reward_anti_hack_multiple_boxed():
    # Dumping many boxed answers to game exact-match -> no reward.
    assert correctness_reward("\\boxed{1}\\boxed{2}\\boxed{18}", "18") == 0.0


def test_correctness_reward_repeated_same_boxed_ok():
    # Same value repeated is one distinct value -> allowed.
    assert correctness_reward("\\boxed{18} ... \\boxed{18}", "18") == 1.0


def test_format_reward_single_boxed():
    assert format_reward("answer \\boxed{18}") == 0.1


def test_format_reward_no_boxed():
    assert format_reward("the answer is 18") == 0.0


def test_format_reward_multiple_boxed():
    assert format_reward("\\boxed{1} \\boxed{2}") == 0.0


def test_total_reward_capped():
    r = total_reward("\\boxed{18}", "18")
    assert r == 1.1


def test_total_reward_format_only():
    assert total_reward("\\boxed{17}", "18") == 0.1


def test_trl_reward_funcs_batch_signature():
    correctness_func, format_func = make_trl_reward_funcs()
    completions = ["\\boxed{18}", "\\boxed{17}"]
    golds = ["18", "18"]
    assert correctness_func(prompts=["q", "q"], completions=completions, gold_answer=golds) == [1.0, 0.0]
    assert format_func(prompts=["q", "q"], completions=completions, gold_answer=golds) == [0.1, 0.1]

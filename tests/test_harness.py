from grpo_env.eval.harness import score_completion, pass_at_1, pass_at_k, evaluate


def test_score_completion():
    assert score_completion("\\boxed{18}", "18") is True
    assert score_completion("\\boxed{17}", "18") is False


def test_pass_at_1():
    records = [
        {"completion": "\\boxed{18}", "gold_answer": "18"},
        {"completion": "\\boxed{0}", "gold_answer": "5"},
    ]
    assert pass_at_1(records) == 0.5


def test_pass_at_k_any_correct():
    grouped = [
        [{"completion": "\\boxed{1}", "gold_answer": "18"},
         {"completion": "\\boxed{18}", "gold_answer": "18"}],  # one correct
        [{"completion": "\\boxed{1}", "gold_answer": "5"},
         {"completion": "\\boxed{2}", "gold_answer": "5"}],    # none correct
    ]
    assert pass_at_k(grouped) == 0.5


def test_evaluate_with_fake_generate():
    examples = [
        {"prompt": "p1", "gold_answer": "18"},
        {"prompt": "p2", "gold_answer": "5"},
    ]

    def fake_generate(prompts, n):
        # returns n completions per prompt
        table = {"p1": ["\\boxed{18}"], "p2": ["\\boxed{4}"]}
        return [table[p] * n for p in prompts]

    result = evaluate(fake_generate, examples, k=1)
    assert result["pass@1"] == 0.5
    assert result["n"] == 2
    assert len(result["per_example"]) == 2
    assert result["pass@k"] == 0.5

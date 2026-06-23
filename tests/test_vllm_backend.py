import pytest
from grpo_env.eval import vllm_backend


def test_greedy_requires_n_equals_1(monkeypatch):
    # Avoid loading a real model: stub the LLM class.
    class FakeOutput:
        def __init__(self, text):
            self.outputs = [type("O", (), {"text": text})()]

    class FakeLLM:
        def __init__(self, *a, **k):
            pass
        def generate(self, prompts, sampling_params):
            return [FakeOutput("\\boxed{1}") for _ in prompts]

    monkeypatch.setattr(vllm_backend, "LLM", FakeLLM)
    monkeypatch.setattr(vllm_backend, "SamplingParams", lambda **k: k)

    gen = vllm_backend.make_generate_fn("fake-model", temperature=0.0)
    with pytest.raises(AssertionError):
        gen(["p1"], n=4)  # greedy + n>1 must error

    out = gen(["p1", "p2"], n=1)
    assert out == [["\\boxed{1}"], ["\\boxed{1}"]]

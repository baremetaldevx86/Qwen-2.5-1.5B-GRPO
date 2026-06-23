from vllm import LLM, SamplingParams
from typing import Callable


def make_generate_fn(
    model_path: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    seed: int = 0,
) -> Callable[[list[str], int], list[list[str]]]:
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, seed=seed)

    def generate_fn(prompts: list[str], n: int) -> list[list[str]]:
        if temperature == 0.0:
            assert n == 1, "greedy decoding (temperature=0) requires n=1"
        sampling = SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        results = llm.generate(prompts, sampling)
        out = []
        for r in results:
            out.append([o.text for o in r.outputs])
        return out

    return generate_fn

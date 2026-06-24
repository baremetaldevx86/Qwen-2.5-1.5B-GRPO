from collections.abc import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_generate_fn(
    model_path: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    seed: int = 0,
    batch_size: int = 8,
    **kwargs,
) -> Callable[[list[str], int], list[list[str]]]:
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    def generate_fn(prompts: list[str], n: int) -> list[list[str]]:
        if temperature == 0.0:
            assert n == 1, "greedy decoding (temperature=0) requires n=1"

        all_completions: list[list[str]] = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            gen_kwargs: dict = dict(
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=n,
            )
            if temperature == 0.0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

            # output_ids shape: (batch_size * n, seq_len)
            new_ids = output_ids[:, prompt_len:]
            decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            # group by prompt
            for j in range(len(batch)):
                all_completions.append(decoded[j * n : (j + 1) * n])

        return all_completions

    return generate_fn

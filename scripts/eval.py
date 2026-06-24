import argparse
import json
import os
import yaml

from grpo_env.utils.seeding import set_seed
from grpo_env.envs.gsm8k import GSM8KEnv
from grpo_env.eval.vllm_backend import make_generate_fn
from grpo_env.eval.harness import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    set_seed(cfg["seed"])
    examples = GSM8KEnv().load(cfg["split"])
    if cfg.get("limit"):
        examples = examples[: cfg["limit"]]

    generate_fn = make_generate_fn(
        cfg["model_path"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        tensor_parallel_size=cfg["tensor_parallel_size"],
        seed=cfg["seed"],
    )
    result = evaluate(generate_fn, examples, k=cfg["k"])
    result["model"] = cfg["model_path"]
    result["tag"] = cfg["tag"]

    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], f"eval_{cfg['tag']}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{cfg['tag']}] pass@1={result['pass@1']:.4f} pass@{cfg['k']}={result['pass@k']:.4f} -> {out_path}")


if __name__ == "__main__":
    main()

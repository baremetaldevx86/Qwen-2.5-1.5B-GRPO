import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAGS = ["base", "sft", "grpo"]
LABELS = {"base": "Base (Qwen2.5-1.5B)", "sft": "+ SFT", "grpo": "+ GRPO"}


def load(tag):
    with open(f"outputs/eval_{tag}.json") as f:
        return json.load(f)


def main():
    results = {tag: load(tag) for tag in TAGS}
    rows = ["| Model | GSM8K test pass@1 |", "|-------|-------------------|"]
    for tag in TAGS:
        rows.append(f"| {LABELS[tag]} | {results[tag]['pass@1']*100:.1f}% |")
    table = "\n".join(rows)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/results_table.md", "w") as f:
        f.write(table + "\n")
    print(table)

    plt.figure()
    plt.bar([LABELS[t] for t in TAGS], [results[t]["pass@1"] * 100 for t in TAGS])
    plt.ylabel("GSM8K test pass@1 (%)")
    plt.title("GRPO post-training lift")
    plt.tight_layout()
    plt.savefig("outputs/pass1_bar.png", dpi=150)

    lift = (results["grpo"]["pass@1"] - results["sft"]["pass@1"]) * 100
    print(f"\nGRPO lift over SFT: +{lift:.1f} pts")


if __name__ == "__main__":
    main()

"""Aggregates per-architecture eval_results_<arch>.json files (produced by
running evaluate.py once per arch) into a single comparison table + plots.
This is the ablation study referenced in docs/system_design.md section 2
and written up in full in docs/RESULTS.md.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MODEL_REGISTRY

BASE = os.path.join(os.path.dirname(__file__), "..")
LOGS_DIR = os.path.join(BASE, "logs")
OUT_DIR = os.path.join(LOGS_DIR, "plots")


def load_results():
    results = {}
    for arch in MODEL_REGISTRY:
        path = os.path.join(LOGS_DIR, f"eval_results_{arch}.json")
        if os.path.isfile(path):
            with open(path) as f:
                results[arch] = json.load(f)
    return results


def print_markdown_table(results):
    headers = ["arch", "hex exact-match", "decimal exact-match", "char acc",
               "params", "size (MB)", "mean latency (ms)", "p95 latency (ms)"]
    rows = []
    for arch, r in results.items():
        rows.append([
            arch,
            f"{r['hex_exact_match_accuracy']:.4f}",
            f"{r['decimal_exact_match_accuracy']:.4f}",
            f"{r['mean_char_accuracy']:.4f}",
            f"{r['parameter_count']:,}",
            f"{r['checkpoint_size_mb']:.3f}",
            f"{r['mean_latency_ms']:.4f}",
            f"{r['p95_latency_ms']:.4f}",
        ])
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))


def plot_comparison(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    archs = list(results.keys())
    accs = [results[a]["hex_exact_match_accuracy"] for a in archs]
    params = [results[a]["parameter_count"] for a in archs]
    latencies = [results[a]["mean_latency_ms"] for a in archs]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    bars0 = axes[0].bar(archs, accs, color=["tab:blue", "tab:orange", "tab:green"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Test exact-match accuracy")
    for b, v in zip(bars0, accs):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")

    bars1 = axes[1].bar(archs, params, color=["tab:blue", "tab:orange", "tab:green"])
    axes[1].set_ylim(0, max(params) * 1.15)
    axes[1].set_title("Parameter count")
    for b, v in zip(bars1, params):
        axes[1].text(b.get_x() + b.get_width() / 2, v + max(params) * 0.02, f"{v:,}", ha="center", fontsize=8)

    bars2 = axes[2].bar(archs, latencies, color=["tab:blue", "tab:orange", "tab:green"])
    axes[2].set_ylim(0, max(latencies) * 1.15)
    axes[2].set_title("Mean latency (ms, batch=1, GPU)")
    for b, v in zip(bars2, latencies):
        axes[2].text(b.get_x() + b.get_width() / 2, v + max(latencies) * 0.02, f"{v:.3f}", ha="center")

    fig.suptitle("Architecture ablation: CRNN (BiGRU) vs. FCN (dilated conv) vs. ConvAttn (self-attention)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "architecture_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"\nsaved {path}")


def main():
    results = load_results()
    if not results:
        print("No eval_results_<arch>.json files found under logs/. "
              "Run evaluate.py --arch <arch> for each trained architecture first.")
        return
    print_markdown_table(results)
    plot_comparison(results)

    with open(os.path.join(LOGS_DIR, "experiment_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

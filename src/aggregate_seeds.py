"""Aggregates per-seed test-set results into mean +- std per architecture,
answering "is the headline accuracy real or n=400/single-seed noise."

Expects weights/model_<arch>_seed<seed>.pt checkpoints (from train.py) to
already exist for each arch/seed pair; runs evaluate.py against each on the
test split, then aggregates.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MODEL_REGISTRY

BASE = os.path.join(os.path.dirname(__file__), "..")
WEIGHTS_DIR = os.path.join(BASE, "weights")
LOGS_DIR = os.path.join(BASE, "logs")


def evaluate_seed(arch: str, seed: int) -> dict:
    ckpt = os.path.join(WEIGHTS_DIR, f"model_{arch}_seed{seed}.pt")
    if not os.path.isfile(ckpt):
        return None
    out_json = os.path.join(LOGS_DIR, "seed_results", f"eval_{arch}_seed{seed}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "evaluate.py"),
        "--arch", arch, "--checkpoint", ckpt, "--split", "test",
        "--out-json", out_json,
    ]
    subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__), check=True)
    with open(out_json) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--out-json", default=os.path.join(LOGS_DIR, "multi_seed_summary.json"))
    args = parser.parse_args()

    summary = {}
    for arch in MODEL_REGISTRY:
        accs = []
        per_seed = {}
        for seed in args.seeds:
            result = evaluate_seed(arch, seed)
            if result is None:
                continue
            accs.append(result["hex_exact_match_accuracy"])
            per_seed[seed] = result["hex_exact_match_accuracy"]
        if not accs:
            continue
        summary[arch] = {
            "seeds_used": list(per_seed.keys()),
            "per_seed_accuracy": per_seed,
            "mean": statistics.mean(accs),
            "std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
            "min": min(accs),
            "max": max(accs),
            "n_seeds": len(accs),
        }

    print(json.dumps(summary, indent=2))
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Plot: per-seed accuracy scatter + mean +- std bar, per architecture.
    archs = list(summary.keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, arch in enumerate(archs):
        seed_accs = list(summary[arch]["per_seed_accuracy"].values())
        ax.scatter([i] * len(seed_accs), seed_accs, color="gray", zorder=2, label="per-seed" if i == 0 else None)
        ax.errorbar(i, summary[arch]["mean"], yerr=summary[arch]["std"], fmt="o", color="tab:blue",
                    capsize=6, markersize=8, zorder=3, label="mean +- std" if i == 0 else None)
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(archs)
    ax.set_ylabel("test hex exact-match accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Multi-seed test accuracy (n={len(args.seeds)} seeds per architecture)")
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(LOGS_DIR, "plots", "multi_seed_accuracy.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()

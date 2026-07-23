"""Parses logs/train_log_<arch>.txt + logs/eval_results_<arch>.json and
renders training curves as PNGs under logs/plots/. Not part of the graded
deliverable list, but useful supporting evidence that the SFT loop actually
converges, and (with --compare) that the architecture ablation is real."""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MODEL_REGISTRY

BASE = os.path.join(os.path.dirname(__file__), "..")
LOGS_DIR = os.path.join(BASE, "logs")
OUT_DIR = os.path.join(LOGS_DIR, "plots")

LINE_RE = re.compile(
    r"epoch (\d+)/\d+ \| train_ctc_loss=([\d.]+) \| val_exact_match_acc=([\d.]+)"
)


def parse_log(path):
    epochs, losses, accs = [], [], []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                accs.append(float(m.group(3)))
    return epochs, losses, accs


def plot_single_arch(arch: str):
    log_path = os.path.join(LOGS_DIR, f"train_log_{arch}.txt")
    eval_path = os.path.join(LOGS_DIR, f"eval_results_{arch}.json")
    if not os.path.isfile(log_path):
        print(f"skip {arch}: no {log_path}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    epochs, losses, accs = parse_log(log_path)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(epochs, losses, color="tab:red")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train CTC loss", color="tab:red")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(epochs, accs, color="tab:blue")
    ax2.set_ylabel("val exact-match accuracy", color="tab:blue")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle(f"[{arch}] SFT training: CTC loss (log) vs. val exact-match accuracy")
    fig.tight_layout()
    curve_path = os.path.join(OUT_DIR, f"training_curves_{arch}.png")
    fig.savefig(curve_path, dpi=150)
    print(f"saved {curve_path}")

    if os.path.isfile(eval_path):
        with open(eval_path) as f:
            results = json.load(f)
        metrics = {
            "hex exact-match": results["hex_exact_match_accuracy"],
            "decimal exact-match": results["decimal_exact_match_accuracy"],
            "mean char accuracy": results["mean_char_accuracy"],
        }
        fig2, ax = plt.subplots(figsize=(5.5, 4))
        bars = ax.bar(metrics.keys(), metrics.values(), color=["tab:blue", "tab:green", "tab:orange"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy")
        ax.set_title(f"[{arch}] Test-set metrics (n={results['n_samples']})")
        for bar, val in zip(bars, metrics.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center")
        fig2.tight_layout()
        bar_path = os.path.join(OUT_DIR, f"test_metrics_{arch}.png")
        fig2.savefig(bar_path, dpi=150)
        print(f"saved {bar_path}")


def plot_val_acc_comparison():
    """Overlay validation exact-match accuracy curves for every architecture
    that has a train_log_<arch>.txt, so convergence speed/plateau is
    directly comparable in one figure."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    any_data = False
    colors = {"crnn": "tab:blue", "fcn": "tab:orange", "convattn": "tab:green"}
    for arch in MODEL_REGISTRY:
        log_path = os.path.join(LOGS_DIR, f"train_log_{arch}.txt")
        if not os.path.isfile(log_path):
            continue
        epochs, _losses, accs = parse_log(log_path)
        if not epochs:
            continue
        ax.plot(epochs, accs, label=arch, color=colors.get(arch))
        any_data = True

    if not any_data:
        print("skip comparison plot: no train_log_<arch>.txt files found")
        return

    ax.set_xlabel("epoch")
    ax.set_ylabel("val exact-match accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation accuracy by architecture (same data, same epoch budget)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "val_accuracy_by_arch.png")
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=None, choices=list(MODEL_REGISTRY),
                         help="plot a single architecture's curves; default plots all found + a comparison overlay")
    args = parser.parse_args()

    if args.arch:
        plot_single_arch(args.arch)
    else:
        for arch in MODEL_REGISTRY:
            plot_single_arch(arch)
        plot_val_acc_comparison()


if __name__ == "__main__":
    main()

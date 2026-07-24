"""Auto-retry training wrapper: guarantees a working, non-collapsed model
regardless of the ~40% per-seed CTC-collapse failure rate documented in
docs/RESULTS.md "Multi-seed robustness" for CRNN/FCN.

Rather than a training-recipe fix (four tried, all net-negative -- see
that section), this is the pragmatic engineering answer: screen each seed
cheaply (a short epoch budget), reject collapsed screens early, and only
pay for a full training run once a seed clears a basic health check. With
an observed ~40% collapse rate, 5 independent attempts leaves roughly a
1% chance every one collapses (0.4^5 ~= 1%).

Usage:
    python train_robust.py --arch crnn --base-seed 42 --max-attempts 5
"""
import argparse
import json
import os
import re
import subprocess
import sys

BASE = os.path.join(os.path.dirname(__file__), "..")
SCREEN_WEIGHTS = os.path.join(BASE, "weights", "robust_screen")
SCREEN_LOGS = os.path.join(BASE, "logs", "robust_screen")

BEST_VAL_RE = re.compile(r"best_val_exact_match_acc=([\d.]+)")


def run_screen(arch: str, seed: int, screen_epochs: int) -> float:
    """Short, cheap training run used only to detect collapse early."""
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "train.py"),
        "--arch", arch, "--seed", str(seed),
        "--epochs", str(screen_epochs), "--patience", "0",  # no early stop during screening
        "--out-dir", SCREEN_WEIGHTS, "--log-dir", SCREEN_LOGS,
    ]
    subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__), check=True)
    log_path = os.path.join(SCREEN_LOGS, f"train_log_{arch}_seed{seed}.txt")
    with open(log_path) as f:
        contents = f.read()
    match = BEST_VAL_RE.search(contents)
    return float(match.group(1)) if match else 0.0


def run_full(arch: str, seed: int, epochs: int, patience: int, out_dir: str, log_dir: str) -> float:
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "train.py"),
        "--arch", arch, "--seed", str(seed),
        "--epochs", str(epochs), "--patience", str(patience),
        "--out-dir", out_dir, "--log-dir", log_dir,
    ]
    subprocess.run(cmd, cwd=os.path.dirname(__file__), check=True)
    log_path = os.path.join(log_dir, f"train_log_{arch}_seed{seed}.txt")
    with open(log_path) as f:
        contents = f.read()
    match = BEST_VAL_RE.search(contents)
    return float(match.group(1)) if match else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=["crnn", "fcn", "convattn"])
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--screen-epochs", type=int, default=20)
    parser.add_argument("--screen-threshold", type=float, default=0.08,
                         help="reject the seed if screening val accuracy is below this")
    parser.add_argument("--full-epochs", type=int, default=80)
    parser.add_argument("--full-patience", type=int, default=12)
    parser.add_argument("--out-dir", default=os.path.join(BASE, "weights"))
    parser.add_argument("--log-dir", default=os.path.join(BASE, "logs"))
    args = parser.parse_args()

    os.makedirs(SCREEN_WEIGHTS, exist_ok=True)
    os.makedirs(SCREEN_LOGS, exist_ok=True)

    attempts = []
    accepted_seed = None

    for i in range(args.max_attempts):
        seed = args.base_seed + i
        screen_acc = run_screen(args.arch, seed, args.screen_epochs)
        passed = screen_acc >= args.screen_threshold
        attempts.append({"seed": seed, "screen_val_acc": screen_acc, "passed": passed})
        print(f"[attempt {i+1}/{args.max_attempts}] seed={seed} "
              f"screen_val_acc={screen_acc:.4f} ({'PASS' if passed else 'reject, retrying'})")
        if passed:
            accepted_seed = seed
            break

    result = {"arch": args.arch, "attempts": attempts, "accepted_seed": accepted_seed}

    if accepted_seed is None:
        print(f"All {args.max_attempts} attempts collapsed at screening -- no seed accepted.")
        result["full_val_acc"] = None
    else:
        print(f"Seed {accepted_seed} passed screening; running full training "
              f"({args.full_epochs} epochs, patience {args.full_patience})...")
        full_acc = run_full(args.arch, accepted_seed, args.full_epochs, args.full_patience,
                             args.out_dir, args.log_dir)
        result["full_val_acc"] = full_acc
        print(f"Full training complete: val_acc={full_acc:.4f}")

    out_json = os.path.join(args.log_dir, f"train_robust_{args.arch}.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Summary written to {out_json}")


if __name__ == "__main__":
    main()

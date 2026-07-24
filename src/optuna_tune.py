"""Light hyperparameter search over learning rate and batch size for the
deployed CRNN, using Optuna. Deliberately small in scope (15 trials, short
epoch budget per trial) -- the assessment brief explicitly says not to
chase accuracy for this PoC, so this exists to demonstrate the technique
and report an honest tuned-vs-untuned comparison, not to squeeze out
maximum accuracy.

Each trial is a real subprocess call to train.py (not an in-process
refactor) writing to a scratch directory, so this file stays a thin
orchestration layer over the same training code used everywhere else in
the repo -- no separate/duplicated training logic to keep in sync.

After the search, src/train_tuned.py-equivalent step (see README) retrains
the best config with the full epoch budget + early stopping for the final
comparison table in docs/RESULTS.md.
"""
import argparse
import json
import os
import re
import subprocess
import sys

BASE = os.path.join(os.path.dirname(__file__), "..")
SCRATCH_WEIGHTS = os.path.join(BASE, "weights", "optuna_scratch")
SCRATCH_LOGS = os.path.join(BASE, "logs", "optuna_scratch")

BEST_VAL_RE = re.compile(r"best_val_exact_match_acc=([\d.]+)")


def run_trial(lr: float, batch_size: int, seed: int, epochs: int, patience: int) -> float:
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "train.py"),
        "--arch", "crnn",
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--out-dir", SCRATCH_WEIGHTS,
        "--log-dir", SCRATCH_LOGS,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))

    # train.py's final "best_val_exact_match_acc=..." summary line is only
    # ever written to the log file, not printed to stdout -- read it there
    # instead of the (empty, for this line) subprocess stdout.
    log_path = os.path.join(SCRATCH_LOGS, f"train_log_crnn_seed{seed}.txt")
    if not os.path.isfile(log_path):
        print(result.stdout[-2000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"train.py did not produce expected log file {log_path}")

    with open(log_path) as f:
        log_contents = f.read()
    match = BEST_VAL_RE.search(log_contents)
    if not match:
        print(log_contents[-2000:], file=sys.stderr)
        raise RuntimeError("Could not parse best_val_exact_match_acc from train.py log file")
    return float(match.group(1))


def main():
    import optuna

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--trial-epochs", type=int, default=25,
                         help="short epoch budget per trial, for search speed")
    parser.add_argument("--trial-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", default=os.path.join(BASE, "logs", "optuna_results.json"))
    args = parser.parse_args()

    os.makedirs(SCRATCH_WEIGHTS, exist_ok=True)
    os.makedirs(SCRATCH_LOGS, exist_ok=True)

    def objective(trial: "optuna.Trial") -> float:
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        val_acc = run_trial(lr, batch_size, args.seed, args.trial_epochs, args.trial_patience)
        return val_acc

    study = optuna.create_study(direction="maximize", study_name="hex_crnn_lr_batch")
    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))

    results = {
        "n_trials": args.n_trials,
        "trial_epochs": args.trial_epochs,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ],
    }
    print(json.dumps(results, indent=2))
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest: {study.best_params} -> val_acc={study.best_value:.4f}")
    print("Retrain with the full budget using these params to get the tuned comparison entry, e.g.:")
    print(f"  python train.py --arch crnn --lr {study.best_params['lr']} "
          f"--batch-size {study.best_params['batch_size']} --seed {args.seed} --epochs 80 --patience 12")


if __name__ == "__main__":
    main()

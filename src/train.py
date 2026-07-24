"""SFT training loop for the HexCRNN recognizer. No RL here by design --
see docs/system_design.md section 0/3 for why RL is out of scope for the
code and specified only in the design doc.

Checkpoints/logs are always written with the seed in the filename
(model_<arch>_seed<seed>.pt, train_log_<arch>_seed<seed>.txt) so multiple
seeds per architecture (see src/multi_seed_eval.py) don't overwrite each
other. When --seed is the canonical default (42), a second, unsuffixed copy
is also written (model_<arch>.pt, train_log_<arch>.txt) since that's the
path every other script (evaluate.py, api.py, compare_experiments.py, ...)
defaults to.
"""
import argparse
import os
import shutil
import time

import torch
from torch.utils.data import DataLoader

from common import BLANK_IDX, ctc_greedy_decode, hex_to_decimal
from dataset import HexImageDataset, collate_fn
from model import MODEL_REGISTRY, count_parameters, get_model

CANONICAL_SEED = 42


def evaluate_split(model, loader, device):
    model.eval()
    correct_hex = 0
    total = 0
    with torch.no_grad():
        for images, targets, target_lengths, hex_texts, _names in loader:
            images = images.to(device)
            log_probs = model(images)
            preds = ctc_greedy_decode(log_probs.cpu())
            for pred, gt in zip(preds, hex_texts):
                if pred == gt:
                    correct_hex += 1
                total += 1
    model.train()
    return correct_hex / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--arch", default="crnn", choices=list(MODEL_REGISTRY),
                         help="which architecture to train (crnn/fcn/convattn) -- see model.py docstring")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--norm", default="batch", choices=["batch", "group"],
                         help="normalization layer in the CNN stem. GroupNorm ('group') was "
                              "tried as a root-cause fix for the seed-dependent CTC collapse -- "
                              "unlike BatchNorm its statistics don't depend on batch composition, "
                              "so it isn't destabilized by an unlucky fresh init. See "
                              "docs/RESULTS.md 'Multi-seed robustness' for the tested result.")
    parser.add_argument("--grad-clip", type=float, default=5.0,
                         help="max gradient norm. A tighter 1.0 was tried as a mitigation for "
                              "seed-dependent CTC mode-collapse, but produced mixed results across "
                              "a 5-seed sweep (helped the worst outlier, hurt several previously-good "
                              "seeds) -- see docs/RESULTS.md 'Multi-seed robustness' for the full "
                              "before/after data. Kept at the original 5.0 default.")
    parser.add_argument("--seed", type=int, default=CANONICAL_SEED)
    parser.add_argument("--patience", type=int, default=8,
                         help="stop if val exact-match accuracy doesn't improve for this many epochs; 0 disables early stopping")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                         help="linear LR warmup from ~0 to --lr over this many epochs before cosine "
                              "decay starts. Tried (with --grad-clip 1.0) as a mitigation for "
                              "seed-dependent CTC mode-collapse; see --grad-clip's help for why it "
                              "was not adopted as the default. 0 disables warmup (original behavior).")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "weights"))
    parser.add_argument("--log-dir", default=os.path.join(os.path.dirname(__file__), "..", "logs"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.abspath(args.data_dir)

    train_set = HexImageDataset(data_dir, "train")
    val_set = HexImageDataset(data_dir, "val")
    # Private generator for the shuffling DataLoader so a given --seed always
    # produces the same batch order regardless of how many random draws
    # happen during model construction (which differs by architecture) --
    # otherwise two archs given the "same seed" can still see different
    # training batch order. Doesn't fix the seed-collapse issue itself
    # (see docs/RESULTS.md), just removes a confound from comparing seeds.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0, generator=loader_generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model = get_model(args.arch, norm=args.norm).to(device)
    n_params = count_parameters(model)
    criterion = torch.nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.05, end_factor=1.0, total_iters=args.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - args.warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    norm_suffix = "" if args.norm == "batch" else f"_{args.norm}norm"
    best_path = os.path.join(args.out_dir, f"model_{args.arch}{norm_suffix}_seed{args.seed}.pt")
    log_path = os.path.join(args.log_dir, f"train_log_{args.arch}{norm_suffix}_seed{args.seed}.txt")

    log_lines = [
        f"arch={args.arch} norm={args.norm} device={device} params={n_params:,} train_samples={len(train_set)} "
        f"val_samples={len(val_set)} epochs={args.epochs} batch_size={args.batch_size} "
        f"lr={args.lr} seed={args.seed} patience={args.patience} warmup_epochs={args.warmup_epochs}"
    ]
    print(log_lines[-1])

    best_val_acc = -1.0
    epochs_since_improvement = 0
    stopped_early_at = None

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for images, targets, target_lengths, _hex_texts, _names in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            log_probs = model(images)  # (T, B, C)
            input_lengths = torch.full(
                (images.size(0),), log_probs.size(0), dtype=torch.long, device=device
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(1, n_batches)
        val_acc = evaluate_split(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        line = (
            f"epoch {epoch:03d}/{args.epochs} | train_ctc_loss={avg_loss:.4f} | "
            f"val_exact_match_acc={val_acc:.4f} | lr={scheduler.get_last_lr()[0]:.6f} | "
            f"time={epoch_time:.1f}s"
        )
        print(line)
        log_lines.append(line)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            epochs_since_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "arch": args.arch,
                    "norm": args.norm,
                    "epoch": epoch,
                    "val_exact_match_acc": val_acc,
                    "seed": args.seed,
                },
                best_path,
            )
        else:
            epochs_since_improvement += 1
            if args.patience > 0 and epochs_since_improvement >= args.patience:
                stopped_early_at = epoch
                log_lines.append(
                    f"early stopping at epoch {epoch}: no val improvement for {args.patience} epochs"
                )
                print(log_lines[-1])
                break

    log_lines.append(f"best_val_exact_match_acc={best_val_acc:.4f} saved_to={best_path}")
    if stopped_early_at:
        log_lines.append(f"stopped_early_at_epoch={stopped_early_at}")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"Training complete. Best checkpoint: {best_path} (val_acc={best_val_acc:.4f})")
    print(f"Log written to {log_path}")

    if args.seed == CANONICAL_SEED:
        # Every other script (evaluate.py, api.py, compare_experiments.py,
        # plot_results.py, error_analysis.py) defaults to the unsuffixed
        # path -- keep it in sync with the canonical-seed run.
        canonical_path = os.path.join(args.out_dir, f"model_{args.arch}.pt")
        canonical_log = os.path.join(args.log_dir, f"train_log_{args.arch}.txt")
        shutil.copyfile(best_path, canonical_path)
        shutil.copyfile(log_path, canonical_log)
        print(f"Also saved canonical copy to {canonical_path}")

        if args.arch == "crnn":
            # The FastAPI service (api.py) loads weights/model.pt by default
            # -- keep the CRNN (the deployed architecture, see
            # docs/RESULTS.md for why it was selected) available there too.
            deployed_path = os.path.join(args.out_dir, "model.pt")
            shutil.copyfile(best_path, deployed_path)
            print(f"Also saved deployment copy to {deployed_path}")


if __name__ == "__main__":
    main()

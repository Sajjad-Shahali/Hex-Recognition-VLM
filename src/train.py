"""SFT training loop for the HexCRNN recognizer. No RL here by design --
see docs/system_design.md section 0/3 for why RL is out of scope for the
code and specified only in the design doc.
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from common import BLANK_IDX, ctc_greedy_decode, hex_to_decimal
from dataset import HexImageDataset, collate_fn
from model import MODEL_REGISTRY, count_parameters, get_model


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "weights"))
    parser.add_argument("--log-path", default=None,
                         help="defaults to logs/train_log_<arch>.txt")
    args = parser.parse_args()
    if args.log_path is None:
        args.log_path = os.path.join(os.path.dirname(__file__), "..", "logs", f"train_log_{args.arch}.txt")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.abspath(args.data_dir)

    train_set = HexImageDataset(data_dir, "train")
    val_set = HexImageDataset(data_dir, "val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model = get_model(args.arch).to(device)
    n_params = count_parameters(model)
    criterion = torch.nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    log_lines = [
        f"arch={args.arch} device={device} params={n_params:,} train_samples={len(train_set)} "
        f"val_samples={len(val_set)} epochs={args.epochs} batch_size={args.batch_size} "
        f"lr={args.lr} seed={args.seed}"
    ]
    print(log_lines[-1])

    best_val_acc = -1.0
    best_path = os.path.join(args.out_dir, f"model_{args.arch}.pt")

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "arch": args.arch,
                    "epoch": epoch,
                    "val_exact_match_acc": val_acc,
                },
                best_path,
            )

    log_lines.append(f"best_val_exact_match_acc={best_val_acc:.4f} saved_to={best_path}")
    with open(args.log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"Training complete. Best checkpoint: {best_path} (val_acc={best_val_acc:.4f})")
    print(f"Log written to {args.log_path}")

    if args.arch == "crnn":
        # The FastAPI service (api.py) loads weights/model.pt by default --
        # keep the CRNN (the deployed architecture, see docs/RESULTS.md for
        # why it was selected) available under that fixed path too. Copy the
        # *best* checkpoint, not the final-epoch weights.
        import shutil
        deployed_path = os.path.join(args.out_dir, "model.pt")
        shutil.copyfile(best_path, deployed_path)
        print(f"Also saved deployment copy to {deployed_path}")


if __name__ == "__main__":
    main()

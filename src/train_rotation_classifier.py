"""Training loop for the 4-way orientation classifier (rotation_model.py).
Standard cross-entropy classification, not CTC -- a separate, much simpler
training loop than train.py's, reflecting the fact that this is a genuinely
different task (orientation detection, not sequence recognition)."""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from rotation_dataset import RotationDataset
from rotation_model import RotationClassifier, count_parameters


def evaluate_split(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data_rotation"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--out-path", default=os.path.join(os.path.dirname(__file__), "..", "weights", "rotation_classifier.pt"))
    parser.add_argument("--log-path", default=os.path.join(os.path.dirname(__file__), "..", "logs", "train_log_rotation_classifier.txt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.abspath(args.data_dir)

    train_set = RotationDataset(data_dir, "train")
    val_set = RotationDataset(data_dir, "val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = RotationClassifier().to(device)
    n_params = count_parameters(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    log_lines = [
        f"params={n_params:,} train_samples={len(train_set)} val_samples={len(val_set)} "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} seed={args.seed}"
    ]
    print(log_lines[-1])

    best_val_acc = -1.0
    epochs_since_improvement = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        running_loss, n_batches = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        val_acc = evaluate_split(model, val_loader, device)
        line = (
            f"epoch {epoch:03d}/{args.epochs} | train_loss={running_loss/max(1,n_batches):.4f} | "
            f"val_acc={val_acc:.4f} | time={time.time()-start:.1f}s"
        )
        print(line)
        log_lines.append(line)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            epochs_since_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc}, args.out_path)
        else:
            epochs_since_improvement += 1
            if args.patience > 0 and epochs_since_improvement >= args.patience:
                log_lines.append(f"early stopping at epoch {epoch}")
                print(log_lines[-1])
                break

    log_lines.append(f"best_val_acc={best_val_acc:.4f} saved_to={args.out_path}")
    with open(args.log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Done. Best checkpoint: {args.out_path} (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()

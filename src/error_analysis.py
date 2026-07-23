"""Error analysis beyond aggregate accuracy: exact-match broken down by
digit length, a character-level confusion matrix, malformed-output rate,
and a montage of actual failure cases. Diagnostic depth the brief's
"proper metrics" language invites but doesn't strictly require -- run after
evaluate.py to understand *how* a model is wrong, not just how often.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from common import VOCAB, ctc_greedy_decode, hex_to_decimal
from dataset import HexImageDataset, collate_fn
from model import MODEL_REGISTRY, get_model

BASE = os.path.join(os.path.dirname(__file__), "..")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(BASE, "data"))
    parser.add_argument("--arch", default=None, choices=list(MODEL_REGISTRY) + [None])
    parser.add_argument("--checkpoint", default=os.path.join(BASE, "weights", "model.pt"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-failure-examples", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.abspath(args.data_dir)

    dataset = HexImageDataset(data_dir, args.split)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    arch = args.arch or checkpoint.get("arch", "crnn")
    model = get_model(arch).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    vocab_index = {c: i for i, c in enumerate(VOCAB)}
    confusion = np.zeros((len(VOCAB), len(VOCAB)), dtype=np.int64)

    by_length_total = {1: 0, 2: 0, 3: 0}
    by_length_correct = {1: 0, 2: 0, 3: 0}
    malformed_count = 0
    total = 0
    failures = []  # (image_name, pred, gt)

    with torch.no_grad():
        for images, targets, target_lengths, hex_texts, names in loader:
            images_gpu = images.to(device)
            log_probs = model(images_gpu)
            preds = ctc_greedy_decode(log_probs.cpu())

            for pred, gt, name, img_tensor in zip(preds, hex_texts, names, images):
                total += 1
                digit_len = len(gt) - 2  # strip "0x"
                by_length_total[digit_len] = by_length_total.get(digit_len, 0) + 1

                if pred == gt:
                    by_length_correct[digit_len] = by_length_correct.get(digit_len, 0) + 1
                else:
                    failures.append((name, pred, gt))

                try:
                    hex_to_decimal(pred)
                except ValueError:
                    malformed_count += 1

                # Character-level confusion, aligned by position up to the
                # shorter string -- an approximation (CTC output isn't
                # position-aligned to ground truth by construction), useful
                # as a directional diagnostic, not a rigorous per-glyph score.
                for p_ch, g_ch in zip(pred, gt):
                    if p_ch in vocab_index and g_ch in vocab_index:
                        confusion[vocab_index[g_ch], vocab_index[p_ch]] += 1

    by_length_accuracy = {
        length: (by_length_correct[length] / by_length_total[length] if by_length_total[length] else None)
        for length in by_length_total
    }

    summary = {
        "arch": arch,
        "split": args.split,
        "n_samples": total,
        "malformed_rate": malformed_count / total,
        "exact_match_by_digit_length": {
            f"{length}_digit": {
                "accuracy": by_length_accuracy[length],
                "n_samples": by_length_total[length],
            }
            for length in sorted(by_length_total)
        },
        "n_failures": len(failures),
    }

    out_json = os.path.join(BASE, "logs", f"error_analysis_{arch}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"saved {out_json}")

    plots_dir = os.path.join(BASE, "logs", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(VOCAB)))
    ax.set_xticklabels(VOCAB, fontsize=8)
    ax.set_yticks(range(len(VOCAB)))
    ax.set_yticklabels(VOCAB, fontsize=8)
    ax.set_xlabel("predicted character")
    ax.set_ylabel("ground-truth character")
    ax.set_title(f"[{arch}] Character confusion matrix ({args.split}, position-aligned approx.)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    cm_path = os.path.join(plots_dir, f"confusion_matrix_{arch}.png")
    fig.savefig(cm_path, dpi=150)
    print(f"saved {cm_path}")

    if failures:
        sample = failures[: args.max_failure_examples]
        img_dir = os.path.join(data_dir, "images", args.split)
        scale = 3
        cell_w, cell_h, label_h = 128 * scale, 32 * scale, 20
        cols = 4
        rows = (len(sample) + cols - 1) // cols
        grid_w = cols * (cell_w + 15) + 15
        grid_h = rows * (cell_h + label_h + 15) + 15
        canvas = Image.new("RGB", (grid_w, grid_h), "white")
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        for i, (name, pred, gt) in enumerate(sample):
            img = Image.open(os.path.join(img_dir, name)).convert("RGB")
            img = img.resize((cell_w, cell_h), Image.NEAREST)
            col, row = i % cols, i // cols
            x = 15 + col * (cell_w + 15)
            y = 15 + row * (cell_h + label_h + 15)
            canvas.paste(img, (x, y))
            draw.text((x, y + cell_h + 2), f"pred={pred} gt={gt}", fill="red", font=font)

        fail_path = os.path.join(plots_dir, f"failures_{arch}.png")
        canvas.save(fail_path)
        print(f"saved {fail_path}")


if __name__ == "__main__":
    main()

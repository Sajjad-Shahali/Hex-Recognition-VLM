"""Evaluation script: exact-match accuracy (hex + decimal), character-level
accuracy, parameter count, checkpoint size, and per-image inference latency.

Latency is measured as a dedicated batch=1 benchmark (100 repeated forward
passes on a single real image, after a warm-up), matching what a live
/predict caller actually experiences -- not derived from batched-inference
throughput, which understates single-request latency and, at small sample
counts, produces a percentile with no statistical meaning (see git history
for the earlier, broken version of this function).
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from common import ctc_greedy_decode, hex_to_decimal
from dataset import HexImageDataset, collate_fn
from model import MODEL_REGISTRY, count_parameters, get_model


def char_accuracy(pred: str, gt: str) -> float:
    """Position-wise character match rate, not edit distance -- a cheap
    proxy for 'how close is the string', not a normalized OCR CER. Insertions
    / deletions (length mismatches) are penalized via the max(len) denominator
    rather than aligned, so this is a directional diagnostic, not a
    standardized metric; exact-match accuracy remains the metric that matters
    for the actual task."""
    if not gt:
        return 1.0 if not pred else 0.0
    matches = sum(1 for p, g in zip(pred, gt) if p == g)
    return matches / max(len(pred), len(gt))


def benchmark_latency(model, device, sample_image, n_reps=100, n_warmup=15):
    """Batch=1 latency benchmark on a single real image, repeated n_reps
    times after a warm-up (excluded from stats) to account for cuDNN
    autotune / lazy CUDA init. Returns mean/p50/p95/max in milliseconds.
    """
    single = sample_image[0:1].to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(single)
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies_ms = []
        for _ in range(n_reps):
            start = time.perf_counter()
            model(single)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms.sort()
    n = len(latencies_ms)
    return {
        "mean_latency_ms": sum(latencies_ms) / n,
        "p50_latency_ms": latencies_ms[n // 2],
        "p95_latency_ms": latencies_ms[int(0.95 * (n - 1))],
        "max_latency_ms": latencies_ms[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--arch", default=None, choices=list(MODEL_REGISTRY) + [None],
                         help="model architecture; inferred from the checkpoint's 'arch' field if omitted")
    parser.add_argument("--checkpoint", default=os.path.join(os.path.dirname(__file__), "..", "weights", "model.pt"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latency-reps", type=int, default=100)
    parser.add_argument("--out-json", default=None,
                         help="defaults to logs/eval_results_<arch>.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.abspath(args.data_dir)

    dataset = HexImageDataset(data_dir, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    arch = args.arch or checkpoint.get("arch", "crnn")
    if args.out_json is None:
        args.out_json = os.path.join(os.path.dirname(__file__), "..", "logs", f"eval_results_{arch}.json")
    model = get_model(arch).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n_params = count_parameters(model)
    checkpoint_size_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)

    hex_exact = 0
    decimal_exact = 0
    char_acc_sum = 0.0
    total = 0
    first_batch_images = None

    with torch.no_grad():
        for images, targets, target_lengths, hex_texts, names in loader:
            if first_batch_images is None:
                first_batch_images = images.clone()

            images = images.to(device)
            log_probs = model(images)
            preds = ctc_greedy_decode(log_probs.cpu())
            for pred, gt in zip(preds, hex_texts):
                total += 1
                char_acc_sum += char_accuracy(pred, gt)
                if pred == gt:
                    hex_exact += 1
                try:
                    pred_decimal = hex_to_decimal(pred)
                    gt_decimal = hex_to_decimal(gt)
                    if pred_decimal == gt_decimal:
                        decimal_exact += 1
                except ValueError:
                    pass  # malformed prediction counts as a decimal miss

    latency = benchmark_latency(model, device, first_batch_images, n_reps=args.latency_reps)

    results = {
        "arch": arch,
        "split": args.split,
        "n_samples": total,
        "hex_exact_match_accuracy": hex_exact / total,
        "decimal_exact_match_accuracy": decimal_exact / total,
        "mean_char_accuracy": char_acc_sum / total,
        "parameter_count": n_params,
        "checkpoint_size_mb": round(checkpoint_size_mb, 3),
        "mean_latency_ms": round(latency["mean_latency_ms"], 4),
        "p50_latency_ms": round(latency["p50_latency_ms"], 4),
        "p95_latency_ms": round(latency["p95_latency_ms"], 4),
        "max_latency_ms": round(latency["max_latency_ms"], 4),
        "latency_reps": args.latency_reps,
        "device": str(device),
    }

    print(json.dumps(results, indent=2))
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

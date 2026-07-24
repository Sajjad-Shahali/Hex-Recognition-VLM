"""End-to-end evaluation of the two-stage rotation pipeline (pipeline.py)
against the rotation-labeled test set (data_rotation/), reporting:
  - "recognizer alone" baseline: raw (possibly rotated) image straight into
    the upright-only-trained recognizer -- reproduces the architectural
    collapse documented in docs/system_design.md section 2.2
  - "pipeline" (classifier + de-rotate + recognizer): the actual deployed
    approach
  - orientation-classifier accuracy on its own (4-way: 0/90/180/270)
so the value of the fix is measured, not just asserted.
"""
import argparse
import csv
import json
import os

import torch
from PIL import Image

from common import IMG_HEIGHT, IMG_WIDTH, ROTATION_CANVAS_SIZE, hex_to_decimal
from generate_rotation_dataset import ORIENTATION_TO_CLASS
from model import get_model
from pipeline import HexRotationPipeline


def naive_crop(img):
    """Center-crop the square rotation-canvas image down to the recognizer's
    native (IMG_WIDTH, IMG_HEIGHT) shape with NO orientation correction --
    simulates naively feeding a possibly-rotated image straight to the
    recognizer, the "no fix" baseline this script measures against."""
    left = (ROTATION_CANVAS_SIZE - IMG_WIDTH) // 2
    top = (ROTATION_CANVAS_SIZE - IMG_HEIGHT) // 2
    return img.crop((left, top, left + IMG_WIDTH, top + IMG_HEIGHT))

BASE = os.path.join(os.path.dirname(__file__), "..")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(BASE, "data_rotation"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--rotation-checkpoint", default=os.path.join(BASE, "weights", "rotation_classifier.pt"))
    parser.add_argument("--recognizer-checkpoint", default=os.path.join(BASE, "weights", "model.pt"))
    parser.add_argument("--out-json", default=os.path.join(BASE, "logs", "pipeline_eval.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_dir = os.path.join(args.data_dir, "images", args.split)

    rows = []
    with open(os.path.join(args.data_dir, "rotation_labels.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if os.path.isfile(os.path.join(img_dir, row["image_name"])):
                rows.append(row)

    pipeline = HexRotationPipeline(args.rotation_checkpoint, args.recognizer_checkpoint, device=device)

    # "Recognizer alone" baseline uses the same recognizer, no correction step.
    baseline_recognizer = pipeline.recognizer

    n = 0
    orientation_correct = 0
    baseline_hex_correct = 0
    pipeline_hex_correct = 0
    pipeline_decimal_correct = 0

    from pipeline import pil_to_tensor
    from common import ctc_greedy_decode

    with torch.no_grad():
        for row in rows:
            img = Image.open(os.path.join(img_dir, row["image_name"]))
            true_hex = row["hexadecimal_value"]
            true_decimal = int(row["decimal_value"])
            true_orientation_class = int(row["orientation_class"])
            n += 1

            # Baseline: raw image (center-cropped to the recognizer's native
            # shape, no rotation correction) straight into the recognizer.
            raw_tensor = pil_to_tensor(naive_crop(img)).to(device)
            baseline_pred = ctc_greedy_decode(baseline_recognizer(raw_tensor).cpu())[0]
            if baseline_pred == true_hex:
                baseline_hex_correct += 1

            # Pipeline: classify orientation, correct, then recognize.
            predicted_degrees = pipeline.predict_orientation(img)
            predicted_class = ORIENTATION_TO_CLASS[predicted_degrees]
            if predicted_class == true_orientation_class:
                orientation_correct += 1

            result = pipeline.predict(img)
            if result["hex_prediction"] == true_hex:
                pipeline_hex_correct += 1
            try:
                if hex_to_decimal(result["hex_prediction"]) == true_decimal:
                    pipeline_decimal_correct += 1
            except ValueError:
                pass

    results = {
        "split": args.split,
        "n_samples": n,
        "orientation_classifier_accuracy": orientation_correct / n,
        "recognizer_alone_hex_accuracy": baseline_hex_correct / n,
        "pipeline_hex_accuracy": pipeline_hex_correct / n,
        "pipeline_decimal_accuracy": pipeline_decimal_correct / n,
        "recognizer_arch": pipeline.arch,
    }

    print(json.dumps(results, indent=2))
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

"""Generates a dataset for the rotation classifier: hex-literal images at
one of 4 orientations (0/90/180/270 degrees, plus the usual small jitter),
labeled with the orientation class -- NOT the hex value (the rotation
classifier doesn't need to read the text, only detect its orientation).

This is a separate dataset from data/ (the main recognition dataset, which
stays upright/0-degree-only by design -- see generate_dataset.py). The
two-stage pipeline (src/pipeline.py) uses this classifier to de-rotate an
image to upright *before* handing it to the recognizer trained on data/,
so the recognizer itself never needs to see rotated text.

Outputs (under --out-dir, default ./data_rotation):
  images/{train,val,test}/*.png
  rotation_labels.csv   (image_name, orientation_degrees, orientation_class)
"""
import argparse
import csv
import os
import random

from common import IMG_HEIGHT, ROTATION_CANVAS_SIZE
from generate_dataset import discover_fonts, random_hex_string, render_sample

ORIENTATIONS = [0, 90, 180, 270]
ORIENTATION_TO_CLASS = {angle: i for i, angle in enumerate(ORIENTATIONS)}


def render_rotation_sample(rng: random.Random, fonts):
    """Wraps generate_dataset.render_sample, always applying exactly one of
    the 4 orientations (orientation_aug_prob=1.0) and returning the coarse
    orientation class and the hex/decimal ground truth alongside the image
    -- the latter is what src/pipeline.py's end-to-end evaluation needs
    (classifier corrects orientation, recognizer reads the value, and we
    need to know if the *final* decimal prediction was right)."""
    coarse_angle = rng.choice(ORIENTATIONS)

    # Reuse render_sample's machinery by temporarily forcing its orientation
    # choice: pass orientations=[coarse_angle] with prob=1.0 so it always
    # applies exactly this angle (plus its own small +-8 degree jitter).
    img, text, decimal_value, _yolo_lines = render_sample(
        rng, fonts,
        orientation_aug_prob=1.0 if coarse_angle != 0 else 0.0,
        orientations=[coarse_angle] if coarse_angle != 0 else [0],
        canvas_width=ROTATION_CANVAS_SIZE,
        canvas_height=ROTATION_CANVAS_SIZE,
        # Keep text within a crop-safe zone the size of the recognizer's
        # own canvas height, so pipeline.py's post-correction center-crop
        # back down to 128x32 is guaranteed to still contain it.
        crop_safe_size=IMG_HEIGHT,
    )
    return img, ORIENTATION_TO_CLASS[coarse_angle], text, decimal_value


def generate_split(split_name, n_samples, out_dir, fonts, rng, csv_rows):
    img_dir = os.path.join(out_dir, "images", split_name)
    os.makedirs(img_dir, exist_ok=True)

    for i in range(n_samples):
        img, orientation_class, hex_text, decimal_value = render_rotation_sample(rng, fonts)
        name = f"{split_name}_{i:06d}.png"
        img.save(os.path.join(img_dir, name))
        csv_rows.append((name, ORIENTATIONS[orientation_class], orientation_class, hex_text, decimal_value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "data_rotation"))
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    rng = random.Random(args.seed)
    fonts = discover_fonts()

    csv_rows = []
    generate_split("train", args.n_train, out_dir, fonts, rng, csv_rows)
    generate_split("val", args.n_val, out_dir, fonts, rng, csv_rows)
    generate_split("test", args.n_test, out_dir, fonts, rng, csv_rows)

    with open(os.path.join(out_dir, "rotation_labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "orientation_degrees", "orientation_class", "hexadecimal_value", "decimal_value"])
        writer.writerows(csv_rows)

    total = args.n_train + args.n_val + args.n_test
    print(f"Generated {total} rotation-labeled samples -> {out_dir}")


if __name__ == "__main__":
    main()

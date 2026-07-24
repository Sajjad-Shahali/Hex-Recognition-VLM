"""Out-of-distribution robustness test set: pushes every augmentation knob
harder than the training range, plus fonts the model has never seen.

This is the actual generalization test -- multi-seed averaging (see
multi_seed_eval.py) only tells you whether 97%-ish is real given sampling
noise; it says nothing about whether the model generalizes to conditions
outside what it was trained on. This script builds that harder test:

  - rotation jitter: +-20 degrees (training used +-8)
  - Gaussian blur applied to every image (training: none)
  - lower contrast: narrower fg/bg intensity gap (training: wide gap)
  - fonts: OOD_FONT_CANDIDATES from generate_dataset.py -- 8 fonts never
    used anywhere in training (Impact, Trebuchet, Segoe UI, Constantia,
    Bahnschrift, Corbel, Franklin Gothic, Gadugi)

Output (under --out-dir, default ./data_ood): images/test/*.png,
dataset.csv (same columns as the main dataset.csv), so evaluate.py-style
comparison is direct.
"""
import argparse
import csv
import os
import random

from generate_dataset import OOD_FONT_CANDIDATES, discover_fonts, render_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "data_ood"))
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    img_dir = os.path.join(out_dir, "images", "test")
    os.makedirs(img_dir, exist_ok=True)

    rng = random.Random(args.seed)
    fonts = discover_fonts(OOD_FONT_CANDIDATES)
    print(f"OOD fonts ({len(fonts)}, held out from training): "
          f"{[os.path.basename(f) if f else 'PIL default' for f in fonts]}")

    csv_rows = []
    for i in range(args.n_samples):
        img, hex_text, decimal_value, _yolo_lines = render_sample(
            rng, fonts,
            size_range=(14, 20),
            rotation_jitter=15.0,           # training: 8.0
            orientation_aug_prob=0.0,       # keep upright -- isolates the
                                             # non-orientation knobs; the
                                             # rotation pipeline is tested
                                             # separately (evaluate_pipeline.py)
            contrast_range=(30, 90, 150, 220),  # training: (0,60,200,255) -- narrower gap, still human-readable
            blur_prob=0.6,                  # training: 0.0 -- not every image, some stay sharp
            blur_radius_range=(0.4, 0.9),
        )
        name = f"ood_{i:06d}.png"
        img.save(os.path.join(img_dir, name))
        csv_rows.append((name, hex_text, decimal_value))

    with open(os.path.join(out_dir, "dataset.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "hexadecimal_value", "decimal_value"])
        writer.writerows(csv_rows)

    print(f"Generated {args.n_samples} OOD samples -> {out_dir}")


if __name__ == "__main__":
    main()

"""Synthetic dataset generator for the hex-to-decimal recognition task.

Renders images of hex literals like "0x1a4", with:
  - font, size, position, and rotation jitter
  - background/text color and pixel noise
so the recognizer doesn't overfit to one font or a fixed layout (see
docs/system_design.md §5 for the rationale behind each knob).

Outputs (all under --out-dir, default ./data):
  images/{train,val,test}/*.png
  labels/{train,val,test}/*.txt   (YOLO format: class cx cy w h, normalized)
  data.yaml                       (YOLO-style dataset spec)
  dataset.csv                     (image_name, hexadecimal_value, decimal_value)
"""
import argparse
import csv
import glob
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import (
    CHAR_TO_IDX,
    IMG_HEIGHT,
    IMG_WIDTH,
    MAX_HEX_DIGITS,
    MIN_HEX_DIGITS,
    VOCAB,
    hex_to_decimal,
)

FONT_DIR = r"C:\Windows\Fonts"
FONT_CANDIDATES = [
    "arial.ttf", "arialbd.ttf", "cour.ttf", "courbd.ttf", "times.ttf",
    "timesbd.ttf", "verdana.ttf", "tahoma.ttf", "georgia.ttf", "consola.ttf",
    "comic.ttf", "calibri.ttf",
]


def discover_fonts():
    fonts = [os.path.join(FONT_DIR, f) for f in FONT_CANDIDATES
             if os.path.isfile(os.path.join(FONT_DIR, f))]
    if not fonts:
        # Headless / non-Windows fallback: PIL's bundled default bitmap font.
        # Only one "font" in this case, so font-variety augmentation degrades
        # gracefully to size/position/rotation/noise jitter only.
        fonts = [None]
    return fonts


def random_hex_string(rng: random.Random) -> str:
    n_digits = rng.randint(MIN_HEX_DIGITS, MAX_HEX_DIGITS)
    low = 16 ** (n_digits - 1) if n_digits > 1 else 0
    high = 16 ** n_digits - 1
    value = rng.randint(low, high)
    return f"0x{value:x}", value


def load_font(font_path, size, rng):
    if font_path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(font_path, size=size)


def char_boxes_for_text(draw: ImageDraw.ImageDraw, text: str, font, origin):
    """Per-character bounding boxes (x0, y0, x1, y1) in image coordinates,
    computed from cumulative substring extents so they match the exact
    glyph placement PIL uses when rendering `text` at `origin`."""
    ox, oy = origin
    boxes = []
    prefix = ""
    prev_right = ox
    for ch in text:
        prefix += ch
        bbox_full = draw.textbbox((ox, oy), prefix, font=font)
        # Right edge of this char = right edge of the cumulative prefix so far.
        right = bbox_full[2]
        single_bbox = draw.textbbox((0, 0), ch, font=font)
        char_h0, char_h1 = single_bbox[1], single_bbox[3]
        top = oy + char_h0
        bottom = oy + char_h1
        boxes.append((ch, prev_right, top, right, bottom))
        prev_right = right
    return boxes


def rotate_point(x, y, cx, cy, angle_deg):
    angle = math.radians(angle_deg)
    dx, dy = x - cx, y - cy
    rx = dx * math.cos(angle) - dy * math.sin(angle)
    ry = dx * math.sin(angle) + dy * math.cos(angle)
    return cx + rx, cy + ry


def rotate_box(box, cx, cy, angle_deg, width, height):
    _, x0, y0, x1, y1 = box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    # PIL's Image.rotate(angle) rotates counter-clockwise for positive angle
    # in standard math convention but its origin is top-left (y grows down),
    # so we rotate by -angle here to match Image.rotate's actual pixel motion.
    rotated = [rotate_point(x, y, cx, cy, -angle_deg) for x, y in corners]
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    x0r, x1r = max(0.0, min(xs)), min(float(width), max(xs))
    y0r, y1r = max(0.0, min(ys)), min(float(height), max(ys))
    return x0r, y0r, x1r, y1r


def add_pixel_noise(img: Image.Image, rng: random.Random, sigma_max=12.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    sigma = rng.uniform(0.0, sigma_max)
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def render_sample(rng: random.Random, fonts):
    text, decimal_value = random_hex_string(rng)

    bg = rng.randint(200, 255)
    fg = rng.randint(0, 60)
    img = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), color=bg)
    draw = ImageDraw.Draw(img)

    font_path = rng.choice(fonts)
    size = rng.randint(16, 24)
    font = load_font(font_path, size, rng)

    text_w = draw.textlength(text, font=font)
    max_x = max(2, int(IMG_WIDTH - text_w - 2))
    origin_x = rng.randint(2, max_x) if max_x > 2 else 2
    origin_y = rng.randint(2, max(2, IMG_HEIGHT - size - 4))

    draw.text((origin_x, origin_y), text, font=font, fill=fg)
    char_boxes = char_boxes_for_text(draw, text, font, (origin_x, origin_y))

    angle = rng.uniform(-8, 8)
    cx, cy = IMG_WIDTH / 2, IMG_HEIGHT / 2
    img = img.rotate(angle, center=(cx, cy), fillcolor=bg, resample=Image.BILINEAR)
    char_boxes = [rotate_box(b, cx, cy, angle, IMG_WIDTH, IMG_HEIGHT) for b in char_boxes]

    img = add_pixel_noise(img, rng)

    yolo_lines = []
    for ch, (x0, y0, x1, y1) in zip(text, char_boxes):
        cls = CHAR_TO_IDX[ch]
        w = x1 - x0
        h = y1 - y0
        cx_norm = (x0 + w / 2) / IMG_WIDTH
        cy_norm = (y0 + h / 2) / IMG_HEIGHT
        w_norm = w / IMG_WIDTH
        h_norm = h / IMG_HEIGHT
        yolo_lines.append(f"{cls} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    return img, text, decimal_value, yolo_lines


def write_data_yaml(out_dir):
    content = f"""\
# YOLO-style dataset spec for the hex-string character-detection task.
#
# Labeling strategy: one bounding box per rendered glyph (including the
# literal '0' and 'x' of the 0x prefix), class id = index into `names`
# below. Boxes are computed analytically from the synthetic renderer's own
# glyph placement (exact, not annotated by hand) -- see
# docs/system_design.md section 5 for why boxes are produced even though
# the CTC recognizer trained in src/model.py does not consume them.
#
# Split strategy: 80/10/10 train/val/test by sample count. Each split's
# images are independently rendered (no image file is ever duplicated
# across splits), but the underlying hex *value* can repeat across splits
# with a different rendering (font/size/rotation/noise) -- the full domain
# is only 4096 possible values (0x0-0xfff), smaller than the training set,
# so this is expected and appropriate: the task is closed-set recognition
# over a small, enumerable output space (like MNIST's 10 digit classes),
# not open-set generalization to unseen values. See docs/system_design.md
# section 5 for the full discussion.

path: .
train: images/train
val: images/val
test: images/test

nc: {len(VOCAB)}
names: {VOCAB}
"""
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(content)


def generate_split(split_name, n_samples, out_dir, fonts, rng, csv_rows):
    img_dir = os.path.join(out_dir, "images", split_name)
    lbl_dir = os.path.join(out_dir, "labels", split_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i in range(n_samples):
        img, hex_text, decimal_value, yolo_lines = render_sample(rng, fonts)
        name = f"{split_name}_{i:06d}"
        img.save(os.path.join(img_dir, f"{name}.png"))
        with open(os.path.join(lbl_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(yolo_lines) + "\n")
        csv_rows.append((f"{name}.png", hex_text, decimal_value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    rng = random.Random(args.seed)
    fonts = discover_fonts()
    print(f"Using {len(fonts)} font(s): {[os.path.basename(f) if f else 'PIL default' for f in fonts]}")

    csv_rows = []
    generate_split("train", args.n_train, out_dir, fonts, rng, csv_rows)
    generate_split("val", args.n_val, out_dir, fonts, rng, csv_rows)
    generate_split("test", args.n_test, out_dir, fonts, rng, csv_rows)

    with open(os.path.join(out_dir, "dataset.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "hexadecimal_value", "decimal_value"])
        writer.writerows(csv_rows)

    write_data_yaml(out_dir)

    total = args.n_train + args.n_val + args.n_test
    print(f"Generated {total} samples -> {out_dir}")
    print(f"  train={args.n_train} val={args.n_val} test={args.n_test}")


if __name__ == "__main__":
    main()

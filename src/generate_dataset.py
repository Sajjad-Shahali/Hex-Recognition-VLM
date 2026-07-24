"""Synthetic dataset generator for the hex-to-decimal recognition task.

Renders images of hex literals like "0x1a4", with:
  - font, size, position, and small-angle rotation jitter
  - background/text color and pixel noise
so the recognizer doesn't overfit to one font or layout (see
docs/system_design.md §5 for the rationale behind each knob).

Canvas is common.IMG_WIDTH x IMG_HEIGHT (128x32) -- the recognizer's
original, seed-stable shape. Coarse 90/180/270-degree orientation is NOT
part of this dataset; it's handled entirely by a separate rotation
classifier trained on its own square canvas (see generate_rotation_dataset.py
and src/pipeline.py) after an earlier attempt to train the recognizer
directly on rotated data caused a severe seed-dependent training collapse
(docs/RESULTS.md "Multi-seed robustness").

Outputs (all under --out-dir, default ./data):
  images/{train,val,test}/*.png
  labels/{train,val,test}/*.txt   (YOLO format: class cx cy w h, normalized)
  data.yaml                       (YOLO-style dataset spec)
  dataset.csv                     (image_name, hexadecimal_value, decimal_value)

Also used as a library by generate_ood_dataset.py, which reuses render_sample
with harsher parameters (wider rotation, blur, lower contrast, held-out fonts)
to build an out-of-distribution robustness test set.
"""
import argparse
import csv
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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

# Fonts never used at training time -- reserved for the OOD robustness set
# (generate_ood_dataset.py) so "unseen font" is a genuine held-out condition.
OOD_FONT_CANDIDATES = [
    "impact.ttf", "trebuc.ttf", "segoeui.ttf", "constan.ttf", "bahnschrift.ttf",
    "corbel.ttf", "framd.ttf", "gadugi.ttf",
]

DEFAULT_ORIENTATIONS = [0, 90, 180, 270]


def discover_fonts(candidates=None):
    candidates = candidates if candidates is not None else FONT_CANDIDATES
    fonts = [os.path.join(FONT_DIR, f) for f in candidates
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


def load_font(font_path, size):
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


def render_sample(
    rng: random.Random,
    fonts,
    size_range=(14, 20),
    rotation_jitter=8.0,
    orientation_aug_prob=0.0,
    orientations=None,
    contrast_range=(0, 60, 200, 255),  # (fg_lo, fg_hi, bg_lo, bg_hi)
    blur_prob=0.0,
    blur_radius_range=(0.5, 1.2),
    canvas_width=None,
    canvas_height=None,
    crop_safe_size=None,
):
    """Render one synthetic hex-literal sample.

    canvas_width/canvas_height: default to common.IMG_WIDTH/IMG_HEIGHT (the
    recognizer's 128x32 canvas). generate_rotation_dataset.py overrides
    these to a square canvas (128x128) instead -- coarse 90/180/270-degree
    orientation is only ever needed for training/evaluating the separate
    rotation classifier, never the recognizer itself (see
    docs/system_design.md section 2.2 for why the recognizer stays on a
    stable, shallow, 128x32-native stem and orientation correction happens
    as an upstream pre-processing step instead).

    crop_safe_size: when set (generate_rotation_dataset.py passes
    common.IMG_HEIGHT=32), constrains the safe-margin placement radius to
    this size instead of the full canvas -- so that after src/pipeline.py
    corrects orientation and center-crops the square canvas back down to
    the recognizer's 128x32 shape, the text is guaranteed to still be
    inside that crop window. Without this, position jitter that uses the
    full 128-tall canvas could place text outside a 32-tall center crop.

    orientation_aug_prob: probability of adding a coarse 90/180/270-degree
    rotation on top of the small (+-rotation_jitter) jitter. 0.0 (default)
    for the main recognition dataset; generate_rotation_dataset.py always
    passes a nonzero value. Ground truth label is unchanged regardless of
    orientation.

    contrast_range=(fg_lo, fg_hi, bg_lo, bg_hi): intensity ranges for text
    (fg) and background (bg) pixels. The OOD generator narrows this gap to
    simulate lower-contrast real-world photos.
    """
    canvas_width = canvas_width if canvas_width is not None else IMG_WIDTH
    canvas_height = canvas_height if canvas_height is not None else IMG_HEIGHT
    orientations = orientations if orientations is not None else DEFAULT_ORIENTATIONS
    text, decimal_value = random_hex_string(rng)

    fg_lo, fg_hi, bg_lo, bg_hi = contrast_range
    bg = rng.randint(bg_lo, bg_hi)
    fg = rng.randint(fg_lo, fg_hi)
    img = Image.new("L", (canvas_width, canvas_height), color=bg)
    draw = ImageDraw.Draw(img)

    font_path = rng.choice(fonts)
    size = rng.randint(*size_range)
    font = load_font(font_path, size)

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    # Safe-margin placement: when orientation_aug_prob > 0, the text's
    # bounding circle (half-diagonal) must stay within the canvas after
    # *any* rotation (0/90/180/270 + jitter), so the jitter offset is
    # constrained by how much room is left once that circle is centered on
    # the canvas. For the main (orientation_aug_prob=0) dataset this is
    # unnecessarily conservative -- only the +-rotation_jitter degrees ever
    # gets applied -- so a simple bounded jitter is used instead, matching
    # the original (pre-rotation-support) placement logic and giving back
    # the full position diversity that logic had.
    if orientation_aug_prob > 0:
        half_diag = math.sqrt(text_w ** 2 + text_h ** 2) / 2
        cx_img, cy_img = canvas_width / 2, canvas_height / 2
        margin_dim = crop_safe_size if crop_safe_size is not None else min(canvas_width, canvas_height)
        max_offset = max(0.0, margin_dim / 2 - half_diag - 2)
        offset_x = rng.uniform(-max_offset, max_offset)
        offset_y = rng.uniform(-max_offset, max_offset)
        origin_x = cx_img + offset_x - text_w / 2 - text_bbox[0]
        origin_y = cy_img + offset_y - text_h / 2 - text_bbox[1]
    else:
        max_x = max(2, canvas_width - text_w - 2)
        origin_x = rng.uniform(2, max_x) - text_bbox[0]
        origin_y = rng.uniform(2, max(2, canvas_height - text_h - 2)) - text_bbox[1]

    draw.text((origin_x, origin_y), text, font=font, fill=fg)
    char_boxes = char_boxes_for_text(draw, text, font, (origin_x, origin_y))

    jitter_angle = rng.uniform(-rotation_jitter, rotation_jitter)
    coarse_angle = rng.choice([a for a in orientations if a != 0]) if rng.random() < orientation_aug_prob else 0
    total_angle = jitter_angle + coarse_angle

    cx_img, cy_img = canvas_width / 2, canvas_height / 2
    img = img.rotate(total_angle, center=(cx_img, cy_img), fillcolor=bg, resample=Image.BILINEAR)
    char_boxes = [rotate_box(b, cx_img, cy_img, total_angle, canvas_width, canvas_height) for b in char_boxes]

    if blur_prob > 0 and rng.random() < blur_prob:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*blur_radius_range)))

    img = add_pixel_noise(img, rng)

    yolo_lines = []
    for ch, (x0, y0, x1, y1) in zip(text, char_boxes):
        cls = CHAR_TO_IDX[ch]
        w = x1 - x0
        h = y1 - y0
        cx_norm = (x0 + w / 2) / canvas_width
        cy_norm = (y0 + h / 2) / canvas_height
        w_norm = w / canvas_width
        h_norm = h / canvas_height
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
# Orientation augmentation: 30% of samples get an additional 90/180/270-
# degree rotation on top of the usual +-8-degree jitter, so the model is
# trained on genuinely upside-down and sideways text, not only near-upright
# text. Canvas is square (128x128) specifically so this doesn't clip content.
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


def generate_split(split_name, n_samples, out_dir, fonts, rng, csv_rows, **render_kwargs):
    img_dir = os.path.join(out_dir, "images", split_name)
    lbl_dir = os.path.join(out_dir, "labels", split_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i in range(n_samples):
        img, hex_text, decimal_value, yolo_lines = render_sample(rng, fonts, **render_kwargs)
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
    parser.add_argument("--orientation-aug-prob", type=float, default=0.0,
                         help="probability of an extra 90/180/270-degree rotation; "
                              "0.0 (default) keeps the main recognition dataset upright "
                              "(+-8-degree jitter only) -- see generate_rotation_dataset.py "
                              "for the oriented dataset used by the rotation classifier")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    rng = random.Random(args.seed)
    fonts = discover_fonts()
    print(f"Using {len(fonts)} font(s): {[os.path.basename(f) if f else 'PIL default' for f in fonts]}")

    csv_rows = []
    render_kwargs = {"orientation_aug_prob": args.orientation_aug_prob}
    generate_split("train", args.n_train, out_dir, fonts, rng, csv_rows, **render_kwargs)
    generate_split("val", args.n_val, out_dir, fonts, rng, csv_rows, **render_kwargs)
    generate_split("test", args.n_test, out_dir, fonts, rng, csv_rows, **render_kwargs)

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

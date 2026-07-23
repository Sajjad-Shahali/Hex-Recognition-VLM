"""PyTorch Dataset for the hex-recognition task, reading dataset.csv + images
produced by generate_dataset.py."""
import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from common import CHAR_TO_IDX, IMG_HEIGHT, IMG_WIDTH


def encode_text(text: str) -> list:
    return [CHAR_TO_IDX[c] for c in text]


class HexImageDataset(Dataset):
    """Expects the layout produced by generate_dataset.py:
    data/images/{split}/*.png + data/dataset.csv (image_name -> hex string).
    """

    def __init__(self, data_dir: str, split: str, csv_path: str = None):
        self.data_dir = data_dir
        self.split = split
        self.img_dir = os.path.join(data_dir, "images", split)

        csv_path = csv_path or os.path.join(data_dir, "dataset.csv")
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # dataset.csv holds all splits; keep only images that exist
                # in this split's image directory.
                if os.path.isfile(os.path.join(self.img_dir, row["image_name"])):
                    rows.append(row)
        if not rows:
            raise RuntimeError(
                f"No samples found for split={split!r} in {self.img_dir}. "
                f"Did you run generate_dataset.py?"
            )
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = os.path.join(self.img_dir, row["image_name"])
        img = Image.open(img_path).convert("L")
        assert img.size == (IMG_WIDTH, IMG_HEIGHT), f"unexpected image size {img.size}"

        tensor = torch.from_numpy(np.array(img, dtype="float32")).unsqueeze(0)
        tensor = (tensor / 255.0 - 0.5) / 0.5  # normalize to [-1, 1]

        hex_text = row["hexadecimal_value"]
        target = torch.tensor(encode_text(hex_text), dtype=torch.long)

        return tensor, target, hex_text, row["image_name"]


def collate_fn(batch):
    images, targets, hex_texts, names = zip(*batch)
    images = torch.stack(images, dim=0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets)
    return images, targets_concat, target_lengths, hex_texts, names

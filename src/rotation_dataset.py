"""PyTorch Dataset for the rotation-classifier data produced by
generate_rotation_dataset.py."""
import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from common import ROTATION_CANVAS_SIZE


class RotationDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        self.img_dir = os.path.join(data_dir, "images", split)
        csv_path = os.path.join(data_dir, "rotation_labels.csv")

        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if os.path.isfile(os.path.join(self.img_dir, row["image_name"])):
                    rows.append(row)
        if not rows:
            raise RuntimeError(
                f"No samples found for split={split!r} in {self.img_dir}. "
                f"Did you run generate_rotation_dataset.py?"
            )
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(os.path.join(self.img_dir, row["image_name"])).convert("L")
        assert img.size == (ROTATION_CANVAS_SIZE, ROTATION_CANVAS_SIZE), f"unexpected image size {img.size}"

        tensor = torch.from_numpy(np.array(img, dtype="float32")).unsqueeze(0)
        tensor = (tensor / 255.0 - 0.5) / 0.5

        label = int(row["orientation_class"])
        return tensor, label

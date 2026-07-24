"""Two-stage inference pipeline: RotationClassifier predicts the image's
coarse orientation (0/90/180/270) on its own square canvas, the square
image is de-rotated back to upright and center-cropped down to the
recognizer's native 128x32 shape, then the (upright-only-trained,
seed-stable, shallow-stem) hex recognizer reads it.

This exists because a straight CTC recognizer with a height-collapsing CNN
stem cannot read 90/270-degree rotated text at all (see
docs/system_design.md section 2.2) -- rather than redesign the recognizer
architecture (which, when tried, also introduced a severe seed-dependent
training collapse -- see docs/RESULTS.md "Multi-seed robustness"), a small,
cheap, very accurate (99.67% val) classifier corrects orientation upstream
on a separate square canvas, letting the recognizer stay on its original,
reliable 128x32 architecture untouched.
"""
import numpy as np
import torch
from PIL import Image

from common import IMG_HEIGHT, IMG_WIDTH, ROTATION_CANVAS_SIZE, ctc_greedy_decode
from generate_rotation_dataset import ORIENTATIONS
from model import get_model
from rotation_model import RotationClassifier


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("L"), dtype="float32")
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return (tensor / 255.0 - 0.5) / 0.5


class HexRotationPipeline:
    def __init__(self, rotation_ckpt: str, recognizer_ckpt: str, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rot_checkpoint = torch.load(rotation_ckpt, map_location=self.device)
        self.rotation_model = RotationClassifier().to(self.device)
        self.rotation_model.load_state_dict(rot_checkpoint["model_state_dict"])
        self.rotation_model.eval()

        rec_checkpoint = torch.load(recognizer_ckpt, map_location=self.device)
        arch = rec_checkpoint.get("arch", "crnn")
        self.recognizer = get_model(arch, norm=rec_checkpoint.get("norm", "batch")).to(self.device)
        self.recognizer.load_state_dict(rec_checkpoint["model_state_dict"])
        self.recognizer.eval()
        self.arch = arch

    @torch.no_grad()
    def predict_orientation(self, img: Image.Image) -> int:
        """img must already be on the square ROTATION_CANVAS_SIZE canvas
        (generate_rotation_dataset.py produces images in this shape)."""
        tensor = pil_to_tensor(img).to(self.device)
        logits = self.rotation_model(tensor)
        predicted_class = logits.argmax(dim=1).item()
        return ORIENTATIONS[predicted_class]

    def correct_orientation(self, img: Image.Image, predicted_degrees: int) -> Image.Image:
        """De-rotate the square image, then center-crop down to the
        recognizer's native 128x32 canvas. Safe because
        generate_rotation_dataset.py constrains text placement to a
        crop_safe_size=IMG_HEIGHT zone specifically so this crop always
        contains it (see render_sample's crop_safe_size docstring)."""
        if predicted_degrees != 0:
            # Undo the augmentation's own rotation convention by applying
            # its exact inverse (same PIL rotate() call, negated angle) --
            # avoids any sign-convention mismatch with how
            # generate_rotation_dataset.py applied the original rotation.
            cx, cy = ROTATION_CANVAS_SIZE / 2, ROTATION_CANVAS_SIZE / 2
            img = img.rotate(-predicted_degrees, center=(cx, cy), fillcolor=230, resample=Image.BILINEAR)

        # Center-crop the square canvas down to (IMG_WIDTH, IMG_HEIGHT).
        left = (ROTATION_CANVAS_SIZE - IMG_WIDTH) // 2
        top = (ROTATION_CANVAS_SIZE - IMG_HEIGHT) // 2
        return img.crop((left, top, left + IMG_WIDTH, top + IMG_HEIGHT))

    @torch.no_grad()
    def predict(self, img: Image.Image) -> dict:
        predicted_degrees = self.predict_orientation(img)
        corrected = self.correct_orientation(img, predicted_degrees)

        tensor = pil_to_tensor(corrected).to(self.device)
        log_probs = self.recognizer(tensor)
        hex_prediction = ctc_greedy_decode(log_probs.cpu())[0]

        return {
            "predicted_orientation_degrees": predicted_degrees,
            "hex_prediction": hex_prediction,
        }

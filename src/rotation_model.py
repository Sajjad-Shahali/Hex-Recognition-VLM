"""4-way orientation classifier (0/90/180/270 degrees) used as a pre-stage
ahead of the hex recognizer -- see docs/system_design.md section 2.2 for why
a height-collapsing CRNN can't handle 90/270-degree rotated text directly,
and pipeline.py for how this classifier's prediction is used to de-rotate
an image before handing it to the recognizer.

Unlike the recognizer stem (_cnn_stem in model.py), this network keeps 2D
spatial structure through a global average pool instead of collapsing
height early -- orientation is exactly the information a height-collapsing
design would destroy, so this network is deliberately NOT built on top of
the recognizer's shared stem.
"""
import torch
import torch.nn as nn

NUM_ORIENTATIONS = 4  # 0, 90, 180, 270 degrees


class RotationClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_ORIENTATIONS):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> (B, 64, 1, 1), orientation-invariant pooling
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x).flatten(1)  # (B, 64)
        return self.classifier(features)   # (B, num_classes) logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = RotationClassifier()
    dummy = torch.randn(4, 1, 128, 128)
    out = model(dummy)
    print(f"output shape: {tuple(out.shape)} | params: {count_parameters(model):,}")

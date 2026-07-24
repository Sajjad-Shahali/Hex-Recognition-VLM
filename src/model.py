"""Three small recognizer architectures for hex-literal images, all built
from torch.nn primitives (no pretrained backbone), sharing the same CNN
stem, CTC output convention, and parameter budget target (<2M params) --
this is the ablation referenced in docs/system_design.md section 2:

  - HexCRNN     : CNN + BiGRU + CTC        (recurrent baseline)
  - HexFCN      : CNN + dilated Conv1d + CTC  (fully convolutional, no
                  recurrence -- the architecture family from Yousef et al.
                  2018 and Coquenet et al. 2020, see design doc references)
  - HexConvAttn : CNN + self-attention encoder + CTC (the encoder family
                  Hernandez Diaz et al. 2021 found competitive with/better
                  than recurrent encoders for line recognition)

Input:  (B, 1, 32, 128) grayscale image. A square 128x128 canvas (needing a
        deeper, 6-block stem) was tried to support 90-degree orientation
        augmentation directly, but that deeper stem turned out to cause a
        ~40%-of-seeds CTC training collapse -- see docs/RESULTS.md
        "Multi-seed robustness". Reverted to this shallower, seed-stable
        4-block stem; orientation robustness now lives entirely in a
        separate pipeline stage (src/pipeline.py) on its own square canvas.
Output: (T, B, NUM_CLASSES) log-probabilities over the vocabulary + CTC
        blank, T=32 timesteps (one per pixel-column after downsampling),
        ready to feed straight into nn.CTCLoss.
"""
import math

import torch
import torch.nn as nn

from common import NUM_CLASSES


def _norm2d(norm: str, channels: int) -> nn.Module:
    """BatchNorm (default, original) or GroupNorm (opt-in, --norm group in
    train.py). GroupNorm's statistics don't depend on batch composition, so
    it doesn't have BatchNorm's known instability in the first few steps of
    training from a fresh random init -- a plausible contributor to the
    seed-dependent CTC collapse documented in docs/RESULTS.md section 2."""
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
    raise ValueError(f"Unknown norm {norm!r}, expected 'batch' or 'group'")


def _norm1d(norm: str, channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm1d(channels)
    if norm == "group":
        return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
    raise ValueError(f"Unknown norm {norm!r}, expected 'batch' or 'group'")


def _cnn_stem(norm: str = "batch"):
    """Shared feature extractor: (B, 1, 32, 128) -> (B, 128, 1, 32).
    Identical across all three architectures so any accuracy/latency
    difference between them is attributable to the sequence-modeling head,
    not to different visual features. The shallow, 4-block version -- see
    module docstring for why this replaced a deeper 6-block variant.
    """
    return nn.Sequential(
        # (1, 32, 128) -> (32, 16, 64)
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # (32, 16, 64) -> (64, 8, 32)
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # (64, 8, 32) -> (128, 4, 32) -- pool height only, keep width as the sequence axis
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        _norm2d(norm, 128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

        # (128, 4, 32) -> (128, 2, 32)
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        _norm2d(norm, 128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

        # (128, 2, 32) -> (128, 1, 32) -- collapse height to 1
        nn.Conv2d(128, 128, kernel_size=(2, 3), padding=(0, 1)),
        _norm2d(norm, 128),
        nn.ReLU(inplace=True),
    )


class HexCRNN(nn.Module):
    """CNN + BiGRU + CTC -- recurrent baseline."""

    def __init__(self, num_classes: int = NUM_CLASSES, rnn_hidden: int = 128, norm: str = "batch"):
        super().__init__()
        self.cnn = _cnn_stem(norm=norm)
        self.rnn = nn.GRU(
            input_size=128, hidden_size=rnn_hidden, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.classifier = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x).squeeze(2).permute(0, 2, 1)  # (B, T=32, C=128)
        rnn_out, _ = self.rnn(features)
        log_probs = self.classifier(rnn_out).log_softmax(dim=2)
        return log_probs.permute(1, 0, 2)  # (T, B, NUM_CLASSES)


class HexFCN(nn.Module):
    """CNN + dilated Conv1d stack + CTC -- no recurrent connections.

    Context along the width (sequence) axis is built with stacked, dilated
    1D convolutions instead of a BiGRU: dilation 1/2/4 gives each output
    timestep a receptive field of +-7 columns, enough to cover a 5-glyph
    hex literal on a 32-column feature map. Fully parallelizable across the
    sequence axis at both train and inference time, unlike the BiGRU.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 128, norm: str = "batch"):
        super().__init__()
        self.cnn = _cnn_stem(norm=norm)
        self.temporal = nn.Sequential(
            nn.Conv1d(128, hidden, kernel_size=3, padding=1, dilation=1),
            _norm1d(norm, hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            _norm1d(norm, hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            _norm1d(norm, hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv1d(hidden, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x).squeeze(2)     # (B, C=128, T=32)
        features = self.temporal(features)    # (B, hidden, T=32)
        logits = self.classifier(features)    # (B, NUM_CLASSES, T=32)
        log_probs = logits.log_softmax(dim=1).permute(2, 0, 1)  # (T, B, NUM_CLASSES)
        return log_probs


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class HexConvAttn(nn.Module):
    """CNN + self-attention encoder + CTC.

    Replaces the BiGRU with a 2-layer Transformer encoder over the
    32-timestep sequence. No recurrence, and unlike HexFCN's fixed dilated
    receptive field, self-attention lets every timestep attend to every
    other timestep directly (useful if a rotated/jittered glyph's evidence
    ends up spatially far from where it "should" be).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, d_model: int = 128, nhead: int = 4, norm: str = "batch"):
        super().__init__()
        self.cnn = _cnn_stem(norm=norm)
        self.pos_encoding = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x).squeeze(2).permute(0, 2, 1)  # (B, T=32, C=128)
        features = self.pos_encoding(features)
        encoded = self.encoder(features)                    # (B, T=32, C=128)
        log_probs = self.classifier(encoded).log_softmax(dim=2)
        return log_probs.permute(1, 0, 2)  # (T, B, NUM_CLASSES)


MODEL_REGISTRY = {
    "crnn": HexCRNN,
    "fcn": HexFCN,
    "convattn": HexConvAttn,
}


def get_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}, choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy = torch.randn(4, 1, 128, 128)
    for name in MODEL_REGISTRY:
        model = get_model(name)
        out = model(dummy)
        print(f"{name:10s} output shape (T, B, C): {tuple(out.shape)} | params: {count_parameters(model):,}")

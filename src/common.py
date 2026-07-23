"""Shared constants used by the dataset generator, model, training, eval, and API.

Keeping these in one place guarantees the vocabulary/decoding logic used at
inference time exactly matches what the model was trained against.
"""

# Character vocabulary rendered/recognized in every image, e.g. "0x1a4".
# Index position doubles as the CTC class id and the YOLO class id.
VOCAB = list("0123456789abcdefx")
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
BLANK_IDX = len(VOCAB)  # CTC blank token, one past the last real class
NUM_CLASSES = len(VOCAB) + 1  # +1 for CTC blank

IMG_WIDTH = 128
IMG_HEIGHT = 32

MIN_HEX_DIGITS = 1
MAX_HEX_DIGITS = 3


def hex_to_decimal(hex_string: str) -> int:
    """'0x1a4' -> 420. Raises ValueError on malformed input."""
    if not hex_string.lower().startswith("0x"):
        raise ValueError(f"expected 0x-prefixed hex string, got {hex_string!r}")
    return int(hex_string, 16)


def ctc_greedy_decode(log_probs) -> list:
    """Greedy CTC decode: argmax per timestep, collapse repeats, drop blanks.

    log_probs: tensor of shape (T, B, NUM_CLASSES).
    Returns: list of B decoded strings.
    """
    best_path = log_probs.argmax(dim=2).transpose(0, 1)  # (B, T)
    results = []
    for seq in best_path.tolist():
        chars = []
        prev = None
        for idx in seq:
            if idx != prev and idx != BLANK_IDX:
                chars.append(IDX_TO_CHAR[idx])
            prev = idx
        results.append("".join(chars))
    return results

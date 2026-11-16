"""Shared settings for the Speech Commands v2, 12-label experiment."""
import torch

SAMPLE_RATE = 16_000
N_FFT = 400              # 25 ms window
HOP_LENGTH = 160         # 10 ms hop
N_MELS = 40
CLIP_SAMPLES = 16_000
KEYWORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
CLASSES = KEYWORDS + ("unknown", "silence")
NUM_CLASSES = len(CLASSES)


def get_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def frame_end_ms(frame_index):
    """Last audio time seen by a center=False STFT frame, in milliseconds."""
    return (N_FFT + frame_index * HOP_LENGTH) * 1000.0 / SAMPLE_RATE


def prefix_frame_index(milliseconds, total_frames):
    """First frame whose *right edge* is at least the given time."""
    import math
    index = math.ceil((milliseconds * SAMPLE_RATE / 1000.0 - N_FFT) / HOP_LENGTH)
    return max(0, min(total_frames - 1, index))

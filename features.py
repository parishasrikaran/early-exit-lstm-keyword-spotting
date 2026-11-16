"""Identical causal log-mel settings for offline and incoming audio."""
import torch
import torchaudio
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS


def create_mel():
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, center=False, power=2.0
    )


def to_features(waveform, mel=None):
    """Input: [..., audio samples] -> [..., frames, 40]. No future padding."""
    mel = mel if mel is not None else create_mel()
    return torch.log(mel(waveform).clamp_min(1e-6)).transpose(-1, -2)


class CausalMelStream:
    """Accept arbitrary-sized waveform chunks and return ONLY newly available frames.

    Holds samples needed to form the next 25-ms window; no samples are reused
    as if they were new frames. A float32 mono 1-D torch tensor is required.
    """
    def __init__(self):
        self.mel = create_mel()
        self.reset()

    def reset(self):
        self.buffer = torch.empty(0, dtype=torch.float32)
        self.total_frames = 0

    @torch.no_grad()
    def push(self, samples):
        samples = torch.as_tensor(samples, dtype=torch.float32).flatten().cpu()
        if samples.numel() == 0:
            return torch.empty(0, N_MELS)
        self.buffer = torch.cat([self.buffer, samples])
        if self.buffer.numel() < N_FFT:
            return torch.empty(0, N_MELS)
        num_frames = 1 + (self.buffer.numel() - N_FFT) // HOP_LENGTH
        feats = to_features(self.buffer.unsqueeze(0), self.mel).squeeze(0)
        assert feats.size(0) == num_frames
        self.buffer = self.buffer[num_frames * HOP_LENGTH:]
        self.total_frames += num_frames
        return feats

"""Speech Commands v2 with explicit silence and subsampled unknown speech."""
from pathlib import Path
import random

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from config import CLASSES, KEYWORDS, SAMPLE_RATE, CLIP_SAMPLES
from features import create_mel, to_features


class SpeechCommands12(Dataset):
    def __init__(self, subset="training", root="data", unknown_ratio=0.1,
                 silence_ratio=0.1, seed=42, max_items=None, download=True):
        self.subset = subset
        self.seed = seed
        self.mel = create_mel()
        ds = torchaudio.datasets.SPEECHCOMMANDS(
            root=root, url="speech_commands_v0.02", subset=subset, download=download
        )
        paths = list(ds._walker)  # file list from official speaker-aware split
        known = [p for p in paths if Path(p).parent.name in KEYWORDS]
        unknown = [p for p in paths
                   if Path(p).parent.name not in KEYWORDS
                   and Path(p).parent.name != "_background_noise_"]
        rng = random.Random(seed)
        rng.shuffle(unknown)
        n_unknown = min(len(unknown), round(len(known) * unknown_ratio))
        n_silence = round(len(known) * silence_ratio)
        self.items = [(p, CLASSES.index(Path(p).parent.name)) for p in known]
        self.items += [(p, 10) for p in unknown[:n_unknown]]
        self.items += [(None, 11) for _ in range(n_silence)]
        rng.shuffle(self.items)
        if max_items is not None:
            self.items = self.items[:max_items]

        # Ambient samples from the dataset's background noises are used to make
        # silence examples less trivial than a zero waveform.
        noise_dir = Path(ds._path) / "_background_noise_"
        self.noises = []
        if noise_dir.is_dir():
            for path in sorted(noise_dir.glob("*.wav")):
                audio, sr = sf.read(path, dtype="float32", always_2d=True)
                if sr == SAMPLE_RATE:
                    self.noises.append(torch.from_numpy(audio.mean(axis=1).copy()))

    def __len__(self):
        return len(self.items)

    def _silence_wave(self, index):
        if not self.noises:
            return torch.zeros(CLIP_SAMPLES)
        rng = random.Random(self.seed * 100003 + index)
        noise = self.noises[rng.randrange(len(self.noises))]
        if len(noise) < CLIP_SAMPLES:
            return torch.zeros(CLIP_SAMPLES)
        start = rng.randint(0, len(noise) - CLIP_SAMPLES)
        return (noise[start:start + CLIP_SAMPLES] * rng.uniform(0.01, 0.08)).clone()

    def __getitem__(self, index):
        path, class_id = self.items[index]
        if path is None:
            wave = self._silence_wave(index)
        else:
            samples, sr = sf.read(path, dtype="float32", always_2d=True)
            wave = torch.from_numpy(np.ascontiguousarray(samples.mean(axis=1)))
            if sr != SAMPLE_RATE:
                wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
            wave = wave[:CLIP_SAMPLES]
            if wave.numel() < CLIP_SAMPLES:
                wave = torch.nn.functional.pad(wave, (0, CLIP_SAMPLES - wave.numel()))
        features = to_features(wave.unsqueeze(0), self.mel).squeeze(0)
        return features, class_id

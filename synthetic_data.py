"""Generate a SMALL synthetic-speech dataset to exercise the KWS training pipeline.

This is *not* Google Speech Commands, not human-recorded speech, and not a
scientific KWS benchmark. Separate espeak voices for train/validation/test;
individual utterances use different rate, pitch, position, gain, and noise.
"""
import argparse
import io
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from config import CLASSES, SAMPLE_RATE, CLIP_SAMPLES
from features import create_mel, to_features

VOICE_SPLITS = {
    "train": ["en", "en-us", "en-gb", "en-uk-wmids"],
    "validation": ["en-uk-rp", "en-wi"],
    "test": ["en-sc", "en-uk-north"],
}
UNKNOWN = ["cat", "dog", "tree", "five", "book", "bird", "red", "blue", "two", "six", "zero", "three"]


def synthesize(word, voice, speed, pitch):
    command = ["espeak", "-v", voice, "-s", str(speed), "-p", str(pitch),
               "--stdout", word]
    completed = subprocess.run(command, capture_output=True, check=True)
    audio, sr = sf.read(io.BytesIO(completed.stdout), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        from math import gcd
        g = gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
    # Do not stretch waveform from the future. Remove end-silence only, allowing
    # onsets to retain their natural TTS timing. Split by voice at dataset level.
    return audio.astype(np.float32)


def crop_augment(wave, rng):
    wave = wave.copy()
    # Isolated one-second clip. Most words fit at chosen TTS speeds.
    start = int(rng.integers(0, 1100))  # 0-69 ms of silence before speech
    out = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    usable = min(len(wave), CLIP_SAMPLES - start)
    out[start:start+usable] = wave[:usable]
    out *= rng.uniform(.55, 1.1)
    if rng.random() < 0.45:
        delay = int(rng.integers(400, 1000))
        out[delay:] += float(rng.uniform(.05, .18)) * out[:-delay]
    # Mild synthetic background; no real-world ambient recordings.
    noise_std = float(rng.uniform(0.0002, 0.004))
    out += rng.normal(0, noise_std, size=CLIP_SAMPLES).astype(np.float32)
    return np.clip(out, -1, 1)


def make_silence(rng):
    output = rng.normal(0, rng.uniform(.0003, .007), size=CLIP_SAMPLES).astype(np.float32)
    if rng.random() < 0.5:
        frequency = rng.uniform(70, 400)
        output += (rng.uniform(.0003, .007)*np.sin(
            2 * np.pi * frequency * np.arange(CLIP_SAMPLES) / SAMPLE_RATE
        )).astype(np.float32)
    return output


def build(args):
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)
    # Deterministic per-split, approximately class-balanced generation.
    mel = create_mel()
    output = {}
    base_cache = {}
    counts = {"train": args.train_per_class_voice,
              "validation": args.valid_per_class_voice,
              "test": args.test_per_class_voice}
    for split, voices in VOICE_SPLITS.items():
        clips, labels, examples = [], [], []
        for voice in voices:
            for class_id, label in enumerate(CLASSES):
                for j in range(counts[split]):
                    if label == "silence":
                        clip = make_silence(rng)
                    else:
                        word = label if label != "unknown" else UNKNOWN[(j + voices.index(voice) * 3) % len(UNKNOWN)]
                        # Alternate pitches and speeds so augmentation is not
                        # just repeated Gaussian noise over a single waveform.
                        variant = j % 6
                        speed = [145, 160, 175, 190, 205, 220][variant]
                        pitch = [32, 46, 57, 67, 40, 52][variant]
                        key = (voice, word, speed, pitch)
                        if key not in base_cache:
                            base_cache[key] = synthesize(word, voice, speed, pitch)
                        clip = crop_augment(base_cache[key], rng)
                    clips.append(clip)
                    labels.append(class_id)
                    if split == "test" and label in ("stop", "yes") and j == 0:
                        examples.append((voice, label, clip.copy()))
        order = list(range(len(labels)))
        py_rng.shuffle(order)
        # Batch feature computation avoids re-reading files every epoch.
        features = []
        with torch.inference_mode():
            for first in range(0, len(order), 64):
                batch = torch.from_numpy(np.stack([clips[k] for k in order[first:first+64]]))
                features.append(to_features(batch, mel).cpu())
        x = torch.cat(features)
        y = torch.tensor([labels[k] for k in order], dtype=torch.long)
        output[split] = {"features": x, "labels": y}
        print(split, len(labels), 'features',tuple(x.shape), flush=True)
        if split == "test":
            Path(args.example_dir).mkdir(parents=True, exist_ok=True)
            for voice, label, clip in examples:
                sf.write(Path(args.example_dir) / f"tts_{label}_{voice}.wav", clip, SAMPLE_RATE)
    output['metadata'] = {
        'dataset': 'Synthetic TTS speech generated locally by espeak; NOT real recorded speech',
        'voices': VOICE_SPLITS, 'classes': list(CLASSES),
        'generation_seed': args.seed,
        'counts_per_class_per_voice': counts,
        'limitations': 'Voice splits are disjoint, but all voices use the same TTS engine. No real acoustic speakers, no environmental audio, not comparable to Speech Commands v2.'
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    Path('results').mkdir(exist_ok=True)
    Path('results/synthetic_metadata.json').write_text(json.dumps(output['metadata'], indent=2))
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/synthetic_tts.pt')
    parser.add_argument('--example-dir', default='examples')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-per-class-voice', type=int, default=18)
    parser.add_argument('--valid-per-class-voice', type=int, default=12)
    parser.add_argument('--test-per-class-voice', type=int, default=12)
    build(parser.parse_args())

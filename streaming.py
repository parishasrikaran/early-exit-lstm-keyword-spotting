"""Incremental LSTM predictions using persistent recurrent state."""
import torch
from early_exit import EarlyExitPolicy
from features import CausalMelStream
from config import SAMPLE_RATE


class StreamingRecognizer:
    def __init__(self, model, device, threshold=0.95, persistence=3,
                 minimum_ms=300, decision_every=2):
        self.model = model.eval()
        self.device = device
        self.features = CausalMelStream()
        self.policy = EarlyExitPolicy(threshold, persistence, minimum_ms, decision_every)
        self.state = None
        self.finished = False

    def reset(self):
        self.features.reset()
        self.policy.reset()
        self.state = None
        self.finished = False

    @torch.inference_mode()
    def push_audio(self, samples):
        """Receive new 16kHz mono samples. Return a decision if one occurs."""
        if self.finished:
            return None
        feats = self.features.push(samples)
        if not feats.size(0):
            return None
        start = self.policy.total_frames
        logits, self.state = self.model(feats.unsqueeze(0).to(self.device), self.state)
        probs = logits[0].softmax(-1).cpu()
        for i, prob in enumerate(probs):
            result = self.policy.update(prob, start + i)
            if result is not None:
                self.finished = True
                return result
        return None


def predict_features_incrementally(model, device, features, threshold=0.95,
                                   persistence=3, minimum_ms=300, decision_every=2,
                                   chunk_frames=2):
    """Run a precomputed clip incrementally, without reprocessing history."""
    model.eval()
    policy = EarlyExitPolicy(threshold, persistence, minimum_ms, decision_every)
    state = None
    with torch.inference_mode():
        for first in range(0, features.size(0), chunk_frames):
            feats = features[first:first + chunk_frames].unsqueeze(0).to(device)
            logits, state = model(feats, state)
            probs = logits[0].softmax(-1).cpu()
            for j, vec in enumerate(probs):
                result = policy.update(vec, first + j)
                if result:
                    return result
    return None


if __name__ == "__main__":
    import argparse
    import soundfile as sf
    import numpy as np
    from config import get_device
    from model import load_checkpoint
    parser = argparse.ArgumentParser(description="Test a 16kHz WAV file in streaming chunks")
    parser.add_argument("wav", help="Path to a 16kHz mono or stereo WAV file")
    parser.add_argument("--checkpoint", default="models/baseline.pt")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--chunk-ms", type=int, default=20)
    args = parser.parse_args()
    device = get_device()
    model, _ = load_checkpoint(args.checkpoint, device)
    wave, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected a 16kHz WAV, got {sr}. Resample it first.")
    mono = wave.mean(axis=1)
    recognizer = StreamingRecognizer(model, device, threshold=args.threshold)
    chunk = int(sr * args.chunk_ms / 1000)
    result = None
    for pos in range(0, len(mono), chunk):
        result = recognizer.push_audio(mono[pos:pos+chunk])
        if result:
            break
    print("Decision:", result if result else "No confident keyword; defer/reject")

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, frame_end_ms
from early_exit import decide_from_probabilities
from features import CausalMelStream, to_features
from model import KeywordLSTM
from streaming import StreamingRecognizer, predict_features_incrementally
from train import compute_loss


def test_mel_chunk_equivalence():
    torch.manual_seed(1)
    waveform = torch.randn(16000) * 0.01
    expected = to_features(waveform.unsqueeze(0)).squeeze(0)
    stream = CausalMelStream()
    parts = []
    for a, b in [(0, 13), (13, 480), (480, 1300), (1300, 5400), (5400, 15993), (15993, 16000)]:
        parts.append(stream.push(waveform[a:b]))
    actual = torch.cat(parts)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_no_future_audio_leakage():
    torch.manual_seed(2)
    original = torch.randn(16000)
    altered = original.clone()
    altered[8000:] *= -1
    a = to_features(original.unsqueeze(0)).squeeze(0)
    b = to_features(altered.unsqueeze(0)).squeeze(0)
    # Frames whose entire 25-ms window ends at/before 500ms must not change.
    num = sum(frame_end_ms(i) <= 500 for i in range(len(a)))
    torch.testing.assert_close(a[:num], b[:num])


def test_threshold_and_persistence():
    scores = np.zeros((98, len(CLASSES)), dtype=np.float32)
    scores[:, 10] = 1.0
    scores[30:38, 0] = .99
    scores[30:38, 10] = .01
    decision = decide_from_probabilities(scores, threshold=.95, persistence=3,
                                         minimum_ms=300, decision_every=2)
    assert decision is not None and decision['label'] == 'yes'
    assert decision['time_ms'] >= 300
    scores[30:38, 0] = .90
    scores[30:38, 10] = .10
    assert decide_from_probabilities(scores, threshold=.95) is None


def test_lstm_full_vs_chunks():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = KeywordLSTM(hidden=16).eval()
    features = torch.randn(1, 98, 40)
    with torch.no_grad():
        logits, _ = model(features)
        state = None
        chunks = []
        for x in features.split(7, dim=1):
            y, state = model(x, state)
            chunks.append(y)
    torch.testing.assert_close(torch.cat(chunks, 1), logits, atol=2e-5, rtol=2e-5)


def test_prefix_loss_computes_gradient():
    logits = torch.randn(3, 98, 12, requires_grad=True)
    targets = torch.tensor([0, 1, 10])
    loss = compute_loss(logits, targets, mode='prefix')
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None

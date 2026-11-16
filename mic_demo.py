"""Mac microphone experiment (one candidate speech event per recording).

This is NOT a continuously listening wake-word detector. Press Enter and speak
one command during a short recording. Uses actual streaming audio chunks and
persistent LSTM state. A TTS-trained checkpoint is not reliable on human speech.
"""
import argparse
import json
import queue
from pathlib import Path

import numpy as np
import torch
from config import SAMPLE_RATE, get_device
from model import load_checkpoint
from streaming import StreamingRecognizer


def run(args):
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit('Install microphone dependency: pip install sounddevice') from exc
    device = get_device('cpu')
    model, _ = load_checkpoint(args.checkpoint, device)
    policy_path = Path(args.policy)
    if policy_path.is_file():
        policy = json.loads(policy_path.read_text())
        threshold = policy['threshold']
        persistence = policy['persistence']
        minimum_ms = policy['minimum_ms']
        decision_every = policy['decision_every']
    else:
        threshold, persistence, minimum_ms, decision_every = args.threshold, 3, 300, 2
    recognizer = StreamingRecognizer(model, device, threshold=threshold,
                                    persistence=persistence, minimum_ms=minimum_ms,
                                    decision_every=decision_every)
    q = queue.Queue()
    blocksize = int(SAMPLE_RATE * args.chunk_ms / 1000)

    def callback(indata, frames, time_info, status):
        if status:
            print('Audio status:', status)
        q.put(indata[:, 0].copy())

    print('WARNING: supplied checkpoints were trained on synthetic TTS voices, not real speakers.')
    print('This is a live-microphone pipeline check only; predictions may be wrong.')
    print('Speak after the recording starts; one event per recording (no automatic VAD).')
    input('Press Enter to record one command: ')
    recognizer.reset()
    result = None
    n_seen = 0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype='float32', blocksize=blocksize, callback=callback,
                        device=args.input_device):
        while n_seen < int(args.duration * SAMPLE_RATE):
            chunk = q.get(timeout=5)
            n_seen += len(chunk)
            result = recognizer.push_audio(chunk)
            if result is not None:
                break
    print('Decision:', result if result else 'DEFER/REJECT — no confident keyword')
    print('Recorded-audio limit:', args.duration, 'seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='models/synthetic_prefix.pt')
    parser.add_argument('--policy', default='results/policy_synthetic_prefix.json')
    parser.add_argument('--threshold', type=float, default=.90)
    parser.add_argument('--chunk-ms', type=int, default=20)
    parser.add_argument('--duration', type=float, default=1.25)
    parser.add_argument('--input-device', type=int, default=None)
    run(parser.parse_args())

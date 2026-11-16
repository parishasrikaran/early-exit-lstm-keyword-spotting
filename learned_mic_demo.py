
import argparse
import json
import queue

import sounddevice as sd

from config import SAMPLE_RATE
from learned_streaming import LearnedStreamingRecognizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="models/real_prefix_full.pt"
    )
    parser.add_argument(
        "--stop-head",
        default="models/stop_head.pt"
    )
    parser.add_argument(
        "--policy",
        default="models/learned_stop_policy.json"
    )
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--chunk-ms", type=int, default=20)
    args = parser.parse_args()

    with open(args.policy) as f:
        policy = json.load(f)

    recognizer = LearnedStreamingRecognizer(
        checkpoint=args.checkpoint,
        stop_head_path=args.stop_head,
        policy=policy,
        device="cpu"
    )

    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        audio_queue.put(indata[:, 0].copy())

    blocksize = int(SAMPLE_RATE * args.chunk_ms / 1000)

    print("Learned StopHead microphone demo ready.")
    print("Press Enter, then speak one command.")

    input()
    recognizer.reset()

    result = None
    samples_seen = 0
    limit = int(args.duration * SAMPLE_RATE)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        callback=callback
    ):
        while samples_seen < limit:
            chunk = audio_queue.get(timeout=5)
            samples_seen += len(chunk)

            result = recognizer.push_audio(chunk)

            if result is not None:
                break

    if result is None:
        print("DEFER/REJECT - no confident keyword")
    else:
        print("Detected keyword:", result["label"])
        print("Confidence:", result["confidence"])
        print("Stop probability:", result["stop_probability"])
        print("Emission time (ms):", result["time_ms"])


if __name__ == "__main__":
    main()

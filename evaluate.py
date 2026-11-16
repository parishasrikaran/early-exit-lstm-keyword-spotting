"""Evaluate checkpoints on the untouched test partition with FROZEN policies."""
import argparse
import json
from pathlib import Path

from config import get_device
from dataset import SpeechCommands12
from train import dataset_for
from metrics import score_probabilities
from model import load_checkpoint
from tune import probabilities_for_dataset


def run(args):
    device = get_device(args.device)
    dataset = dataset_for(args, "test")
    results = {}
    for checkpoint in args.checkpoints:
        name = Path(checkpoint).stem
        policy_path = Path(args.policies_dir) / f"policy_{name}.json"
        if not policy_path.exists():
            raise FileNotFoundError(f"First tune on validation: {policy_path}")
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        model, _ = load_checkpoint(checkpoint, device)
        probs, labels = probabilities_for_dataset(model, device, dataset, args.batch_size)
        scores = score_probabilities(
            probs, labels, policy["threshold"], policy["persistence"],
            policy["minimum_ms"], policy["decision_every"]
        )
        results[name] = {"checkpoint": checkpoint,
                         "policy": str(policy_path), "metrics": scores}
        print("\nModel:", name)
        for metric, value in scores.items():
            print(f"  {metric}: {value}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved:", output)
    print("NOTE: clip scores do not establish real-stream FA/hour or acoustic onset delay.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("speech_commands","synthetic"), default="speech_commands")
    parser.add_argument("--synthetic-path", default="data/synthetic_tts.pt")
    parser.add_argument("--checkpoints", nargs="+", default=["models/baseline.pt", "models/prefix.pt"])
    parser.add_argument("--policies-dir", default="results")
    parser.add_argument("--output", default="results/clip_comparison.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())

"""Choose a confidence threshold on VALIDATION clips, never test clips."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import get_device
from dataset import SpeechCommands12
from train import dataset_for
from metrics import score_probabilities
from model import load_checkpoint


def probabilities_for_dataset(model, device, dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    all_probs, all_labels = [], []
    model.eval()
    with torch.inference_mode():
        for features, labels in loader:
            logits, _ = model(features.to(device))
            probs = logits.softmax(-1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())
    return all_probs, all_labels


def choose_threshold(probabilities, labels, max_negative_trigger_rate=0.02,
                     persistence=3, minimum_ms=300, decision_every=2):
    candidates = []
    for threshold in (0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995):
        metrics = score_probabilities(probabilities, labels, threshold,
                                      persistence, minimum_ms, decision_every)
        false_rate = metrics["early_negative_trigger_rate"]
        recall = metrics["early_keyword_recall"]
        candidates.append((threshold, metrics, false_rate, recall))
    feasible = [c for c in candidates if c[2] is not None
                and c[2] <= max_negative_trigger_rate]
    if feasible:
        # Maximize recognized keywords at the desired false-trigger limit;
        # prefer lower observed p95 emission time when recall ties.
        best = max(feasible, key=lambda c: (c[3] if c[3] is not None else -1,
                        -(c[1]["p95_emission_ms"] or 1e9)))
        status = "validation negative-trigger constraint satisfied"
    else:
        best = min(candidates, key=lambda c: (
            c[2] if c[2] is not None else 1e9,
            -(c[3] if c[3] is not None else -1)))
        status = "WARNING: no threshold met the validation constraint"
    return {"threshold": best[0], "persistence": persistence,
            "minimum_ms": minimum_ms, "decision_every": decision_every,
            "max_negative_trigger_rate": max_negative_trigger_rate,
            "status": status, "validation_metrics": best[1]}


def main(args):
    device = get_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    validation = dataset_for(args, "validation")
    probabilities, labels = probabilities_for_dataset(model, device, validation, args.batch_size)
    chosen = choose_threshold(probabilities, labels, args.max_negative_trigger_rate,
                              args.persistence, args.minimum_ms, args.decision_every)
    output = Path(args.output or f"results/policy_{Path(args.checkpoint).stem}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(json.dumps(chosen, indent=2))
    print("Saved policy:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("speech_commands","synthetic"), default="speech_commands")
    parser.add_argument("--synthetic-path", default="data/synthetic_tts.pt")
    parser.add_argument("--checkpoint", default="models/baseline.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--minimum-ms", type=float, default=300)
    parser.add_argument("--decision-every", type=int, default=2)
    parser.add_argument("--max-negative-trigger-rate", type=float, default=0.02)
    main(parser.parse_args())

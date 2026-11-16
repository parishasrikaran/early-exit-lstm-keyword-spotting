
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import load_checkpoint
from train import dataset_for
from stop_head import StopHead


def collect(model, stop_head, loader, device):
    records = []

    model.eval()
    stop_head.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1)

            batch_size, frames, classes = probs.shape
            streak = torch.zeros(batch_size, device=device)
            previous = None
            predictions = []

            for t in range(30, frames, 4):
                p = probs[:, t, :]
                values, indices = p.topk(2, dim=-1)
                pred = indices[:, 0]

                if previous is None:
                    streak = torch.ones_like(streak)
                else:
                    streak = torch.where(
                        pred == previous,
                        streak + 1,
                        torch.ones_like(streak)
                    )

                previous = pred.clone()

                features = torch.stack([
                    values[:, 0],
                    values[:, 0] - values[:, 1],
                    streak.clamp(max=5) / 5.0,
                    torch.full_like(
                        values[:, 0],
                        min((t + 1) * 0.01, 1.0)
                    )
                ], dim=1)

                scores = stop_head(features).squeeze(1)

                predictions.append((
                    pred.cpu(),
                    scores.cpu(),
                    streak.cpu(),
                    (t + 1) * 10
                ))

            for i in range(batch_size):
                records.append({
                    "label": int(y[i]),
                    "steps": [
                        {
                            "prediction": int(pred[i]),
                            "score": float(score[i]),
                            "streak": float(stable[i]),
                            "time_ms": time_ms
                        }
                        for pred, score, stable, time_ms
                        in predictions
                    ]
                })

    return records


def evaluate(records, threshold):
    positives = 0
    correct = 0
    wrong = 0
    negatives = 0
    negative_triggers = 0
    times = []

    for record in records:
        label = record["label"]
        is_positive = label < 10

        if is_positive:
            positives += 1
        else:
            negatives += 1

        decision = None

        for step in record["steps"]:
            if step["time_ms"] < 300:
                continue
            if step["streak"] < 3:
                continue
            if step["prediction"] >= 10:
                continue
            if step["score"] < threshold:
                continue

            decision = step
            break

        if decision is None:
            continue

        times.append(decision["time_ms"])

        if is_positive:
            if decision["prediction"] == label:
                correct += 1
            else:
                wrong += 1
        else:
            negative_triggers += 1

    return {
        "threshold": threshold,
        "early_keyword_recall": correct / max(positives, 1),
        "early_wrong_keyword_rate": wrong / max(positives, 1),
        "early_negative_trigger_rate":
            negative_triggers / max(negatives, 1),
        "mean_emission_ms":
            sum(times) / len(times) if times else None
    }


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
        "--output",
        default="models/learned_stop_policy.json"
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-valid", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-negative-trigger-rate",
        type=float,
        default=0.02
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    dataset_args = argparse.Namespace(
        dataset="speech_commands",
        synthetic_path="",
        data_dir=args.data_dir,
        seed=args.seed,
        max_train=None,
        max_valid=args.max_valid
    )

    dataset = dataset_for(dataset_args, "validation")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model, _ = load_checkpoint(args.checkpoint, device)

    stop_head = StopHead()
    saved = torch.load(
        args.stop_head,
        map_location=device,
        weights_only=True
    )
    stop_head.load_state_dict(saved["model_state_dict"])

    print("Collecting validation predictions...")
    records = collect(model, stop_head, loader, device)

    candidates = []

    for i in range(50, 100):
        threshold = i / 100
        result = evaluate(records, threshold)
        candidates.append(result)

    valid = [
        r for r in candidates
        if r["early_negative_trigger_rate"]
        <= args.max_negative_trigger_rate
    ]

    if not valid:
        print("No threshold satisfies the constraint.")
        return

    best = max(
        valid,
        key=lambda r: (
            r["early_keyword_recall"],
            -r["early_wrong_keyword_rate"]
        )
    )

    policy = {
        "threshold": best["threshold"],
        "persistence": 3,
        "minimum_ms": 300,
        "decision_every": 4,
        "max_negative_trigger_rate":
            args.max_negative_trigger_rate,
        "validation_metrics": best
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(policy, indent=2))

    print(json.dumps(policy, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()

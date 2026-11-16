
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import load_checkpoint
from stop_head import StopHead
from train import dataset_for
from tune import probabilities_for_dataset
from metrics import score_probabilities


def learned_scores(probs, labels, stop_head, policy):
    threshold = policy["threshold"]
    persistence = policy["persistence"]
    minimum_ms = policy["minimum_ms"]
    decision_every = policy["decision_every"]

    positive = correct = wrong = 0
    negative = negative_triggers = 0
    emitted = 0
    times = []

    stop_head.eval()

    with torch.no_grad():
        for i in range(len(labels)):
            label = int(labels[i])
            p = torch.as_tensor(probs[i]).float()

            if label < 10:
                positive += 1
            else:
                negative += 1

            streak = 0
            previous = None

            for t in range(30, p.shape[0], decision_every):
                values, indices = p[t].topk(2)
                prediction = int(indices[0])

                if prediction == previous:
                    streak += 1
                else:
                    streak = 1

                previous = prediction

                time_ms = (t + 1) * 10

                if time_ms < minimum_ms:
                    continue

                if streak < persistence:
                    continue

                if prediction >= 10:
                    continue

                features = torch.tensor([[
                    float(values[0]),
                    float(values[0] - values[1]),
                    min(streak, 5) / 5.0,
                    min((t + 1) * 0.01, 1.0)
                ]])

                score = float(stop_head(features).item())

                if score < threshold:
                    continue

                emitted += 1
                times.append(time_ms)

                if label < 10:
                    if prediction == label:
                        correct += 1
                    else:
                        wrong += 1
                else:
                    negative_triggers += 1

                break

    return {
        "clips": len(labels),
        "positive_clips": positive,
        "negative_clips": negative,
        "early_keyword_recall": correct / max(positive, 1),
        "early_wrong_keyword_rate": wrong / max(positive, 1),
        "early_miss_rate": 1 - (correct + wrong) / max(positive, 1),
        "early_negative_trigger_rate":
            negative_triggers / max(negative, 1),
        "early_clip_coverage": emitted / max(len(labels), 1),
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
        "--learned-policy",
        default="models/learned_stop_policy.json"
    )
    parser.add_argument(
        "--existing-policy",
        default="models/real_prefix_policy.json"
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="results/learned_stop_comparison.json"
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    dataset_args = argparse.Namespace(
        dataset="speech_commands",
        synthetic_path="",
        data_dir=args.data_dir,
        seed=args.seed,
        max_train=None,
        max_valid=None
    )

    dataset = dataset_for(dataset_args, "test")

    model, _ = load_checkpoint(args.checkpoint, device)

    probs, labels = probabilities_for_dataset(
        model, device, dataset, args.batch_size
    )

    stop_head = StopHead()

    saved = torch.load(
        args.stop_head,
        map_location=device,
        weights_only=True
    )

    stop_head.load_state_dict(saved["model_state_dict"])

    learned_policy = json.loads(
        Path(args.learned_policy).read_text()
    )

    existing_policy = json.loads(
        Path(args.existing_policy).read_text()
    )

    print("Evaluating existing policy...")

    existing = score_probabilities(
        probs,
        labels,
        existing_policy["threshold"],
        existing_policy["persistence"],
        existing_policy["minimum_ms"],
        existing_policy["decision_every"]
    )

    print("Evaluating learned StopHead...")

    learned = learned_scores(
        probs, labels, stop_head, learned_policy
    )

    results = {
        "existing_policy": existing,
        "learned_stop_head": learned
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()

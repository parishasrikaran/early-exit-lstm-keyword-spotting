
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import load_checkpoint
from train import dataset_for
from stop_head import StopHead


def extract_samples(model, loader, device):
    inputs = []
    targets = []

    model.eval()

    with torch.no_grad():
        for batch, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device).long()

            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1)

            previous = None
            streak = torch.zeros(
                x.shape[0], device=device
            )

            for t in range(30, probs.shape[1], 4):
                p = probs[:, t, :]

                values, indices = p.topk(2, dim=-1)
                prediction = indices[:, 0]

                confidence = values[:, 0]
                margin = values[:, 0] - values[:, 1]

                if previous is None:
                    streak = torch.ones_like(streak)
                else:
                    streak = torch.where(
                        prediction == previous,
                        streak + 1,
                        torch.ones_like(streak)
                    )

                previous = prediction.clone()

                stability = streak.clamp(max=5) / 5.0
                elapsed = torch.full_like(
                    confidence,
                    min((t + 1) * 0.01, 1.0)
                )

                features = torch.stack(
                    [confidence, margin, stability, elapsed],
                    dim=1
                )

                # Classes 0-9 are the ten target keywords.
                correct = (
                    (prediction == y) &
                    (prediction < 10)
                ).float()

                inputs.append(features.cpu())
                targets.append(correct.cpu().unsqueeze(1))

            if (batch + 1) % 20 == 0:
                print("Processed batches:", batch + 1)

    return torch.cat(inputs), torch.cat(targets)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="models/real_prefix_full.pt"
    )
    parser.add_argument(
        "--output",
        default="models/stop_head.pt"
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=5000
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--data-dir",
        default="data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    dataset_args = argparse.Namespace(
        dataset="speech_commands",
        synthetic_path="",
        data_dir=args.data_dir,
        seed=args.seed,
        max_train=args.max_train,
        max_valid=None
    )

    dataset = dataset_for(dataset_args, "train")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model, _ = load_checkpoint(
        args.checkpoint, device
    )

    print("Extracting partial-audio predictions...")

    X, Y = extract_samples(model, loader, device)

    print("Training samples:", len(X))
    print("Positive fraction:", Y.mean().item())

    stop_head = StopHead()

    optimizer = torch.optim.Adam(
        stop_head.parameters(),
        lr=0.001
    )

    criterion = torch.nn.BCELoss()

    training_loader = DataLoader(
        torch.utils.data.TensorDataset(X, Y),
        batch_size=256,
        shuffle=True
    )

    for epoch in range(args.epochs):
        stop_head.train()
        total_loss = 0

        for inputs, targets in training_loader:
            optimizer.zero_grad()

            outputs = stop_head(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Loss: {total_loss / len(training_loader):.4f}"
        )

    Path(args.output).parent.mkdir(
        parents=True, exist_ok=True
    )

    torch.save({
        "model_state_dict": stop_head.state_dict(),
        "input_features": [
            "confidence",
            "margin",
            "stability",
            "elapsed_time"
        ],
        "recognition_checkpoint": args.checkpoint
    }, args.output)

    print("Saved StopHead:", args.output)


if __name__ == "__main__":
    main()

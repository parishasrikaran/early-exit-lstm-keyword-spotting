"""Train a full-window baseline or a conservative prefix-supervised model."""
import argparse
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import get_device, prefix_frame_index
from dataset import SpeechCommands12
from model import KeywordLSTM, save_checkpoint


def compute_loss(logits, labels, mode="baseline", prefix_weight=0.2):
    full_loss = nn.functional.cross_entropy(logits[:, -1, :], labels)
    if mode == "baseline":
        return full_loss
    # Use *later* prefixes only. Prefix labels are not automatically reliable
    # in the first few hundred ms of speech (a documented research limitation).
    positions = sorted(set(prefix_frame_index(t, logits.size(1)) for t in (600, 800)))
    partial_losses = [nn.functional.cross_entropy(logits[:, idx, :], labels)
                      for idx in positions if idx < logits.size(1) - 1]
    return full_loss + (prefix_weight * torch.stack(partial_losses).mean()
                        if partial_losses else 0.0)


def dataset_for(args, split):
    if args.dataset == "synthetic":
        source = torch.load(args.synthetic_path, weights_only=True, map_location="cpu")
        return TensorDataset(source[split]["features"], source[split]["labels"])
    name = {"train": "training", "validation": "validation", "test": "testing"}[split]
    max_items = {"train": args.max_train, "validation": args.max_valid, "test": None}[split]
    return SpeechCommands12(name, root=args.data_dir, seed=args.seed, max_items=max_items)

def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    print("Running on", device)
    train_data = dataset_for(args, "train")
    valid_data = dataset_for(args, "validation")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=0)
    print(f"Training examples: {len(train_data)}; validation examples: {len(valid_data)}")
    model = KeywordLSTM(args.hidden, args.layers).to(device)
    if args.dataset == "synthetic":
        mean = train_data.tensors[0].mean(dim=(0, 1))
        std = train_data.tensors[0].std(dim=(0, 1)).clamp_min(.1)
        model.set_normalization(mean, std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    path = args.output or f"models/{args.mode}.pt"
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        seen = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features)
            loss = compute_loss(logits, labels, args.mode, args.prefix_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += loss.item() * len(labels)
            seen += len(labels)
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for features, labels in valid_loader:
                logits, _ = model(features.to(device))
                correct += (logits[:, -1, :].argmax(-1).cpu() == labels).sum().item()
                total += len(labels)
        accuracy = correct / total if total else 0
        history.append({"epoch":epoch, "train_loss":round(loss_total/seen, 5), "validation_accuracy":round(accuracy, 5)})
        print(f"Epoch {epoch:2} | train loss={loss_total/seen:.4f} | validation accuracy={accuracy:.3f}", flush=True)
        if accuracy > best:
            best = accuracy
            save_checkpoint(path, model, epoch, args.mode)
            print("  Saved best checkpoint:", path)
    print(f"Finished. Best validation accuracy: {best:.3f} (not test accuracy)")
    import json
    from pathlib import Path
    report = Path(f"results/training_{args.dataset}_{args.mode}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"dataset":args.dataset, "mode":args.mode, "history":history, "best_validation_accuracy":best, "checkpoint":path}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("speech_commands","synthetic"), default="speech_commands")
    parser.add_argument("--synthetic-path", default="data/synthetic_tts.pt")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--mode", choices=("baseline", "prefix"), default="baseline")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--prefix-weight", type=float, default=0.2)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())

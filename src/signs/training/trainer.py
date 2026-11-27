"""Train the sign classifier and keep the weights with the best validation accuracy."""

import time

import torch
import torch.nn as nn

from signs.config import Config
from signs.data.dataset import load_splits, make_loader
from signs.models.registry import build_model
from signs.runs import CHECKPOINT_FILE, create_run_dir, pick_device, write_csv, write_json


def run_epoch(network, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    network.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = network(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)

    return total_loss / seen, correct / seen


def train(config: Config, overwrite: bool = False) -> dict:
    """Train one run and write its folder. Returns the training summary."""
    run_dir = create_run_dir(config, overwrite=overwrite)
    device = pick_device()
    print(f"run: {run_dir}   device: {device}")

    train_set, val_set, _ = load_splits(config.data)
    train_loader = make_loader(train_set, config.data, shuffle=True)
    val_loader = make_loader(val_set, config.data, shuffle=False)
    print(f"train images: {len(train_set)}   val images: {len(val_set)}")

    network = build_model(config.model).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config.train.lr)

    checkpoint = run_dir / CHECKPOINT_FILE
    history = []
    best = {"epoch": 0, "val_accuracy": 0.0, "val_loss": float("inf")}
    epochs = config.train.epochs

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, train_acc = run_epoch(network, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(network, val_loader, criterion, device)
        seconds = time.perf_counter() - started
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_acc, 6),
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": round(seconds, 2),
            }
        )
        write_csv(run_dir / "history.csv", history)
        print(
            f"epoch {epoch}/{epochs}  "
            f"train loss {train_loss:.3f} acc {train_acc:.3f}  "
            f"val loss {val_loss:.3f} acc {val_acc:.3f}  ({seconds:.0f}s)"
        )
        if val_acc > best["val_accuracy"] or best["epoch"] == 0:
            best = {"epoch": epoch, "val_accuracy": val_acc, "val_loss": val_loss}
            torch.save(
                {"state_dict": network.state_dict(), "val_accuracy": val_acc, "epoch": epoch},
                checkpoint,
            )
            print(f"  saved {checkpoint}")

    summary = {
        "run_name": config.run_name,
        "seed": config.seed,
        "device": str(device),
        "train_images": len(train_set),
        "val_images": len(val_set),
        "best_epoch": best["epoch"],
        "best_val_accuracy": best["val_accuracy"],
        "best_val_loss": best["val_loss"],
    }
    write_json(run_dir / "train_summary.json", summary)
    print(f"best validation accuracy: {best['val_accuracy']:.3f} (epoch {best['epoch']})")
    return summary

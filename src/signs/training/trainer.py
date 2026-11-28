"""Train the sign classifier and keep the weights with the best validation accuracy."""

import time

import torch
import torch.nn as nn

from signs.config import Config, TrainConfig
from signs.data.dataset import load_splits, make_loader
from signs.models.registry import build_model
from signs.runs import CHECKPOINT_FILE, create_run_dir, pick_device, write_csv, write_json
from signs.training.seed import seed_everything
from signs.viz.figures import plot_training_curves


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


def parameter_groups(network: nn.Module, train_config: TrainConfig) -> list[dict]:
    """Trainable parameters, with the pretrained trunk on its own learning rate.

    The trunk is the network's `features` module when it has one. With the
    trunk frozen, only the last layers are returned.
    """
    lr = train_config.lr
    trunk = getattr(network, "features", None)
    trunk_params = [p for p in trunk.parameters() if p.requires_grad] if trunk else []
    trunk_ids = {id(p) for p in trunk_params}
    head_params = [p for p in network.parameters() if p.requires_grad and id(p) not in trunk_ids]
    groups = [{"params": head_params, "lr": lr, "name": "head"}]
    if trunk_params:
        groups.append(
            {"params": trunk_params, "lr": lr * train_config.backbone_lr_scale, "name": "trunk"}
        )
    return groups


def train(config: Config, overwrite: bool = False) -> dict:
    """Train one run and write its folder. Returns the training summary."""
    run_dir = create_run_dir(config, overwrite=overwrite)
    seed_everything(config.seed)
    device = pick_device()
    print(f"run: {run_dir}   device: {device}   seed: {config.seed}")

    train_set, val_set, _ = load_splits(config.data)
    train_loader = make_loader(train_set, config.data, shuffle=True, seed=config.seed)
    val_loader = make_loader(val_set, config.data, shuffle=False)
    print(f"train images: {len(train_set)}   val images: {len(val_set)}")

    network = build_model(config.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        parameter_groups(network, config.train), weight_decay=config.train.weight_decay
    )
    epochs = config.train.epochs
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if config.train.schedule == "cosine"
        else None
    )
    patience = config.train.early_stopping_patience

    checkpoint = run_dir / CHECKPOINT_FILE
    history = []
    best = {"epoch": 0, "val_accuracy": 0.0, "val_loss": float("inf")}
    stopped_early = False

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
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
                "lr": lr,
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
        if scheduler is not None:
            scheduler.step()
        if patience and epoch - best["epoch"] >= patience:
            print(f"  no validation gain for {patience} epochs, stopping")
            stopped_early = True
            break

    summary = {
        "run_name": config.run_name,
        "seed": config.seed,
        "device": str(device),
        "train_images": len(train_set),
        "val_images": len(val_set),
        "best_epoch": best["epoch"],
        "best_val_accuracy": best["val_accuracy"],
        "best_val_loss": best["val_loss"],
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
        "train_seconds": round(sum(row["seconds"] for row in history), 1),
    }
    write_json(run_dir / "train_summary.json", summary)
    plot_training_curves(
        run_dir / "history.csv", run_dir / "training_curves.png", best_epoch=best["epoch"]
    )
    print(f"best validation accuracy: {best['val_accuracy']:.3f} (epoch {best['epoch']})")
    return summary

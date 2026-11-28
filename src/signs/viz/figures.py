"""Figures drawn from the files a run saved, never from live training state."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_training_curves(history_csv: Path, path: Path, best_epoch: int | None = None) -> None:
    """Loss, accuracy and learning rate per epoch, train against validation."""
    with Path(history_csv).open() as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]

    def column(name):
        return [float(row[name]) for row in rows]

    figure, (loss_axis, accuracy_axis, lr_axis) = plt.subplots(1, 3, figsize=(13, 3.6))
    for axis, metric, title in (
        (loss_axis, "loss", "Cross-entropy loss"),
        (accuracy_axis, "accuracy", "Accuracy"),
    ):
        axis.plot(epochs, column(f"train_{metric}"), marker="o", label="train")
        axis.plot(epochs, column(f"val_{metric}"), marker="o", label="validation")
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        if best_epoch:
            axis.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    accuracy_axis.legend()
    lr_axis.plot(epochs, column("lr"), marker="o", color="tab:green")
    lr_axis.set_title("Learning rate (last layer)")
    lr_axis.set_xlabel("epoch")
    lr_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    lr_axis.grid(alpha=0.3)
    if best_epoch:
        figure.suptitle(f"Kept checkpoint: epoch {best_epoch} (dashed line)", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)

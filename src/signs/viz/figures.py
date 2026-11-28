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


def plot_hog_examples(dataset, transform, class_ids: list[int], path: Path) -> None:
    """For a few signs: the photo, the grayscale image HOG reads, and its gradient histograms.

    Each HOG cell is drawn as a star of lines, one per orientation bin, as
    long as that bin is strong. Edges of the sign's outline and symbol show
    up as long strokes along the edge.
    """
    from skimage import exposure

    from signs.class_names import CLASS_NAMES

    first_index = {}
    for index, label in enumerate(dataset.labels):
        if label in class_ids and label not in first_index:
            first_index[label] = index
    missing = [class_id for class_id in class_ids if class_id not in first_index]
    if missing:
        raise ValueError(f"no training image for class(es) {missing}")

    figure, axes = plt.subplots(3, len(class_ids), figsize=(2.2 * len(class_ids), 6.8))
    axes = axes.reshape(3, len(class_ids))
    for column, class_id in enumerate(class_ids):
        # Take a frame from the middle of the track, where the sign is mid-size.
        image, _ = dataset[first_index[class_id] + 15]
        gray = transform.image(image)
        _, drawing = transform(image, visualize=True)
        drawing = exposure.rescale_intensity(drawing, in_range=(0, drawing.max() or 1))
        for row, (picture, cmap) in enumerate(((image, None), (gray, "gray"), (drawing, "gray"))):
            axes[row, column].imshow(picture, cmap=cmap)
            axes[row, column].axis("off")
        axes[0, column].set_title(CLASS_NAMES[class_id], fontsize=8)
    for row, label in enumerate(("photo", "grayscale input", "HOG cells")):
        axes[row, 0].text(
            -0.08,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
        )
    cells = transform.config.pixels_per_cell
    figure.suptitle(
        f"HOG at {transform.image_size}×{transform.image_size}: {cells}×{cells}-pixel cells, "
        f"{transform.config.orientations} orientation bins, "
        f"{transform.feature_size()} features per sign",
        fontsize=10,
    )
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_validation_curves(runs: dict[str, Path], path: Path) -> None:
    """Validation accuracy and loss per epoch for several runs, on shared axes.

    runs maps a legend label to a run folder. A run whose history has no
    epochs (HOG + SVM) is skipped.
    """
    figure, (accuracy_axis, loss_axis) = plt.subplots(1, 2, figsize=(11, 3.8))
    drawn = 0
    for label, run_dir in runs.items():
        with (Path(run_dir) / "history.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "epoch" not in rows[0]:
            continue
        epochs = [int(row["epoch"]) for row in rows]
        accuracy_axis.plot(
            epochs, [float(r["val_accuracy"]) for r in rows], marker="o", label=label
        )
        loss_axis.plot(epochs, [float(r["val_loss"]) for r in rows], marker="o", label=label)
        drawn += 1
    if not drawn:
        plt.close(figure)
        raise ValueError("none of the runs has a per-epoch history")
    accuracy_axis.set_title("Validation accuracy")
    loss_axis.set_title("Validation loss")
    for axis in (accuracy_axis, loss_axis):
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
    accuracy_axis.legend(fontsize=8)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)

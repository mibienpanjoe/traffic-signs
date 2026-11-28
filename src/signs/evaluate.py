"""Score a trained run on its validation tracks and on the official GTSRB test set."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402

from signs.analysis.metrics import (  # noqa: E402
    classification_scores,
    confusion,
    most_confused_pairs,
)
from signs.class_names import CLASS_NAMES  # noqa: E402
from signs.data.dataset import load_splits, make_loader  # noqa: E402
from signs.runs import (  # noqa: E402
    SVM_FILE,
    load_run_config,
    load_trained_model,
    pick_device,
    write_csv,
    write_json,
)

SUMMARY_KEYS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
PAIR_FIELDS = [
    "true_id",
    "true_name",
    "predicted_id",
    "predicted_name",
    "count",
    "share_of_true",
]


def predict_all(network, loader, device):
    truths, predictions = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = network(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            truths.extend(labels.tolist())
    return np.array(truths), np.array(predictions)


def predict_val_and_test(run_dir: Path) -> dict:
    """{"val": (truths, predictions), "test": (truths, predictions)} for a trained run."""
    config = load_run_config(run_dir)
    if config.model.name == "hog_svm":
        from signs.models.hog_svm import HogSVM
        from signs.training.hog_trainer import extract_features

        model = HogSVM.load(Path(run_dir) / SVM_FILE)
        _, val_set, test_set = load_splits(
            config.data, train_tf=model.transform, eval_tf=model.transform
        )
        outputs = {}
        for name, dataset in (("val", val_set), ("test", test_set)):
            features, truths = extract_features(dataset, config.data)
            outputs[name] = (truths, model.predict(features))
        return outputs

    device = pick_device()
    network, config = load_trained_model(run_dir, device)
    _, val_set, test_set = load_splits(config.data)
    outputs = {}
    for name, dataset in (("val", val_set), ("test", test_set)):
        loader = make_loader(dataset, config.data, shuffle=False)
        outputs[name] = predict_all(network, loader, device)
    return outputs


def _round(row: dict) -> dict:
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()}


def plot_confusion(matrix: np.ndarray, accuracy: float, path: Path) -> None:
    """Row-normalized confusion matrix with class names: each row sums to 1."""
    totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, totals, out=np.zeros(matrix.shape), where=totals > 0)
    ticks = [f"{i} {name}" for i, name in enumerate(CLASS_NAMES)]
    figure, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(ticks)), ticks, rotation=90, fontsize=6)
    axis.set_yticks(range(len(ticks)), ticks, fontsize=6)
    axis.set_title(f"Test confusion matrix, share of each true class (accuracy {accuracy:.1%})")
    axis.set_xlabel("predicted class")
    axis.set_ylabel("true class")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def evaluate(run_dir: Path) -> dict:
    """Score validation and test, and write the run's evaluation files.

    metrics.json         accuracy and macro scores, for validation and test
    per_class.csv        test precision, recall, F1 and support per class
    confused_pairs.csv   the most frequent test mistakes, with names
    confusion_matrix.png row-normalized, with names
    classification_report.txt  the per-class test scores, for reading
    """
    run_dir = Path(run_dir)
    config = load_run_config(run_dir)
    results = {
        name: (truths, predictions, classification_scores(truths, predictions))
        for name, (truths, predictions) in predict_val_and_test(run_dir).items()
    }

    truths, predictions, test_scores = results["test"]
    val_scores = results["val"][2]
    matrix = confusion(truths, predictions)
    pairs = most_confused_pairs(matrix, top=15)

    metrics = {
        "run_name": config.run_name,
        "seed": config.seed,
        "split": config.data.split,
        "val": {"images": len(results["val"][0]), **{k: val_scores[k] for k in SUMMARY_KEYS}},
        "test": {"images": len(results["test"][0]), **{k: test_scores[k] for k in SUMMARY_KEYS}},
    }
    write_json(run_dir / "metrics.json", metrics)
    write_csv(run_dir / "per_class.csv", [_round(row) for row in test_scores["per_class"]])
    write_csv(run_dir / "confused_pairs.csv", pairs, fieldnames=PAIR_FIELDS)
    plot_confusion(matrix, test_scores["accuracy"], run_dir / "confusion_matrix.png")
    report = classification_report(
        truths,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=3,
        zero_division=0,
    )
    (run_dir / "classification_report.txt").write_text(
        f"test accuracy: {test_scores['accuracy']:.3f}\n\n{report}"
    )

    print(f"split: {config.data.split}")
    for name in ("val", "test"):
        scores = metrics[name]
        print(
            f"{name:>4}  {scores['images']:>6} images  "
            f"accuracy {scores['accuracy']:.3f}  macro-F1 {scores['macro_f1']:.3f}"
        )
    weakest = sorted(test_scores["per_class"], key=lambda row: row["recall"])[:5]
    print("lowest test recall:")
    for row in weakest:
        print(f"  {row['recall']:.3f}  {row['name']} ({row['support']} images)")
    print("most confused (true -> predicted):")
    for pair in pairs[:5]:
        print(
            f"  {pair['count']:>4}  {pair['true_name']} -> {pair['predicted_name']}"
            f"  ({pair['share_of_true']:.1%} of {pair['true_name']})"
        )
    print(f"saved metrics, per-class scores, confused pairs and confusion matrix in {run_dir}")
    return metrics

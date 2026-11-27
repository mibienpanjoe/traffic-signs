"""Score a trained run on the official GTSRB test set."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from signs.class_names import CLASS_NAMES  # noqa: E402
from signs.data.dataset import load_test_set, make_loader  # noqa: E402
from signs.runs import load_trained_model, pick_device, write_csv, write_json  # noqa: E402


def predict_all(network, loader, device):
    truths, predictions = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = network(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            truths.extend(labels.tolist())
    return np.array(truths), np.array(predictions)


def evaluate(run_dir: Path) -> dict:
    """Write metrics.json, per_class.csv, the text report and the confusion matrix."""
    run_dir = Path(run_dir)
    device = pick_device()
    network, config = load_trained_model(run_dir, device)
    test_set = load_test_set(config.data)
    loader = make_loader(test_set, config.data, shuffle=False)

    truths, predictions = predict_all(network, loader, device)
    labels = list(range(len(CLASS_NAMES)))
    accuracy = float(np.mean(predictions == truths))
    report_kwargs = dict(labels=labels, target_names=CLASS_NAMES, zero_division=0)
    report = classification_report(truths, predictions, digits=3, **report_kwargs)
    scores = classification_report(truths, predictions, output_dict=True, **report_kwargs)

    print(f"test images: {len(test_set)}")
    print(f"test accuracy: {accuracy:.3f}")
    print(report)

    metrics = {
        "run_name": config.run_name,
        "test_images": len(test_set),
        "accuracy": accuracy,
        "macro_precision": scores["macro avg"]["precision"],
        "macro_recall": scores["macro avg"]["recall"],
        "macro_f1": scores["macro avg"]["f1-score"],
    }
    write_json(run_dir / "metrics.json", metrics)
    write_csv(
        run_dir / "per_class.csv",
        [
            {
                "class_id": class_id,
                "name": name,
                "precision": round(scores[name]["precision"], 6),
                "recall": round(scores[name]["recall"], 6),
                "f1": round(scores[name]["f1-score"], 6),
                "support": int(scores[name]["support"]),
            }
            for class_id, name in enumerate(CLASS_NAMES)
        ],
    )
    (run_dir / "classification_report.txt").write_text(f"test accuracy: {accuracy:.3f}\n\n{report}")

    matrix = confusion_matrix(truths, predictions, labels=labels)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(f"Test confusion matrix  (accuracy {accuracy:.1%})")
    axis.set_xlabel("predicted class id")
    axis.set_ylabel("true class id")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(run_dir / "confusion_matrix.png", dpi=120)
    plt.close(figure)
    print(f"saved metrics, per-class scores and confusion matrix in {run_dir}")
    return metrics

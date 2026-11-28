"""Scores computed from true and predicted class ids.

Overall accuracy is dominated by the big classes (Speed limit 30 and 50
have 10 times the images of Speed limit 20), so every score here also
comes per class, and macro averages weigh each class the same.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from signs.class_names import CLASS_NAMES


def classification_scores(truths, predictions, num_classes: int = len(CLASS_NAMES)) -> dict:
    """Accuracy, macro precision/recall/F1, and per-class scores.

    Returns {"accuracy", "macro_precision", "macro_recall", "macro_f1",
    "per_class": [{class_id, name, precision, recall, f1, support}, ...]}.
    A class with no predictions gets precision 0.
    """
    truths = np.asarray(truths)
    predictions = np.asarray(predictions)
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        truths, predictions, labels=labels, zero_division=0
    )
    present = support > 0
    return {
        "accuracy": float(np.mean(truths == predictions)),
        "macro_precision": float(precision[present].mean()),
        "macro_recall": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "per_class": [
            {
                "class_id": class_id,
                "name": CLASS_NAMES[class_id],
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }
            for class_id in labels
        ],
    }


def confusion(truths, predictions, num_classes: int = len(CLASS_NAMES)) -> np.ndarray:
    """Counts with rows = true class, columns = predicted class."""
    return confusion_matrix(truths, predictions, labels=list(range(num_classes)))


def most_confused_pairs(matrix: np.ndarray, top: int = 10) -> list[dict]:
    """The largest off-diagonal cells: which true class is most often taken for which.

    `share_of_true` is the count divided by the true class's images, so a
    rare class that is often wrong still ranks.
    """
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    rows_total = matrix.sum(axis=1)
    order = np.argsort(off_diagonal, axis=None)[::-1]
    pairs = []
    for flat in order[:top]:
        true_id, predicted_id = np.unravel_index(flat, matrix.shape)
        count = int(off_diagonal[true_id, predicted_id])
        if count == 0:
            break
        pairs.append(
            {
                "true_id": int(true_id),
                "true_name": CLASS_NAMES[true_id],
                "predicted_id": int(predicted_id),
                "predicted_name": CLASS_NAMES[predicted_id],
                "count": count,
                "share_of_true": round(count / int(rows_total[true_id]), 6),
            }
        )
    return pairs


def expected_calibration_error(confidences, correct, bins: int = 15) -> float:
    """How far the model's confidence is from its accuracy, on average.

    Predictions are grouped by confidence into equal-width bins. In each
    bin, the gap between mean confidence and accuracy is weighted by the
    bin's share of predictions. A model that says 90% and is right 90% of
    the time scores 0. A model that says 90% and is right 50% of the time
    is overconfident, and its ECE grows.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidences.shape != correct.shape or confidences.size == 0:
        raise ValueError("confidences and correct must be non-empty and the same length")
    # Bin k holds confidences in (k/bins, (k+1)/bins]; a confidence of 0 goes to bin 0.
    which = np.clip(np.ceil(confidences * bins).astype(int) - 1, 0, bins - 1)
    ece = 0.0
    for k in range(bins):
        members = which == k
        if members.any():
            gap = abs(confidences[members].mean() - correct[members].mean())
            ece += members.mean() * gap
    return float(ece)

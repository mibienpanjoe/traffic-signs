import numpy as np
import pytest

from signs.analysis.metrics import classification_scores, confusion, most_confused_pairs


def test_perfect_predictions_score_one():
    truths = [0, 1, 2, 2]
    scores = classification_scores(truths, truths, num_classes=3)
    assert scores["accuracy"] == 1.0
    assert scores["macro_f1"] == 1.0


def test_macro_f1_weighs_a_failed_rare_class():
    # 9 of class 0 right, the single class-1 image wrong: accuracy 0.9.
    truths = [0] * 9 + [1]
    predictions = [0] * 10
    scores = classification_scores(truths, predictions, num_classes=2)
    assert scores["accuracy"] == pytest.approx(0.9)
    assert scores["per_class"][1]["recall"] == 0.0
    # F1 of class 0 is 18/19; class 1 is 0; macro mean is about 0.47.
    assert scores["macro_f1"] == pytest.approx((18 / 19) / 2)


def test_classes_absent_from_the_truth_do_not_lower_the_macro_average():
    scores = classification_scores([0, 0], [0, 0], num_classes=5)
    assert scores["macro_f1"] == 1.0
    assert scores["per_class"][3]["support"] == 0


def test_most_confused_pairs_rank_off_diagonal_counts():
    truths = [1] * 10 + [2] * 10 + [3] * 5
    predictions = [1] * 4 + [2] * 6 + [2] * 9 + [1] + [3] * 5
    pairs = most_confused_pairs(confusion(truths, predictions), top=5)
    assert [(p["true_id"], p["predicted_id"], p["count"]) for p in pairs] == [(1, 2, 6), (2, 1, 1)]
    assert pairs[0]["true_name"] == "Speed limit 30"
    assert pairs[0]["predicted_name"] == "Speed limit 50"
    assert pairs[0]["share_of_true"] == pytest.approx(0.6)


def test_confusion_rows_are_true_classes():
    matrix = confusion([0, 0, 1], [0, 1, 1], num_classes=2)
    assert np.array_equal(matrix, [[1, 1], [0, 1]])

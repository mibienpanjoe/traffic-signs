import json

import pytest

from signs.analysis.aggregate import aggregate_runs, markdown_table, mean_std


def _run(tmp_path, name, seed, val, test, split="track"):
    run_dir = tmp_path / name
    run_dir.mkdir()
    scores = lambda acc: {  # noqa: E731
        "images": 10,
        "accuracy": acc,
        "macro_f1": acc - 0.1,
        "macro_precision": acc,
        "macro_recall": acc,
    }
    metrics = {
        "run_name": name,
        "seed": seed,
        "split": split,
        "val": scores(val),
        "test": scores(test),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    return run_dir


def test_mean_std_uses_the_sample_deviation():
    mean, std = mean_std([0.8, 0.9, 1.0])
    assert mean == pytest.approx(0.9)
    assert std == pytest.approx(0.1)
    assert mean_std([0.5]) == (0.5, 0.0)


def test_aggregate_reports_mean_std_and_gap(tmp_path):
    runs = [
        _run(tmp_path, "a-seed0", 0, val=0.9, test=0.8),
        _run(tmp_path, "a-seed1", 1, val=1.0, test=0.8),
    ]
    row = aggregate_runs(runs, "a")
    assert row["seeds"] == 2
    assert row["test_accuracy_mean"] == pytest.approx(0.8)
    assert row["test_accuracy_std"] == pytest.approx(0.0)
    assert row["val_minus_test_accuracy_mean"] == pytest.approx(0.15)
    assert "| a | track | 2 | 95.0 ± 7.1 | 80.0 ± 0.0 |" in markdown_table([row])


def test_mixed_splits_are_rejected(tmp_path):
    runs = [
        _run(tmp_path, "a-seed0", 0, 0.9, 0.8, split="track"),
        _run(tmp_path, "a-seed1", 1, 0.9, 0.8, split="random"),
    ]
    with pytest.raises(ValueError, match="different splits"):
        aggregate_runs(runs, "a")


def test_repeated_seeds_are_rejected(tmp_path):
    runs = [_run(tmp_path, "a-seed0", 0, 0.9, 0.8), _run(tmp_path, "b-seed0", 0, 0.9, 0.8)]
    with pytest.raises(ValueError, match="repeat a seed"):
        aggregate_runs(runs, "a")


def test_unevaluated_run_is_reported(tmp_path):
    (tmp_path / "a-seed0").mkdir()
    with pytest.raises(FileNotFoundError, match="signs eval"):
        aggregate_runs([tmp_path / "a-seed0"], "a")

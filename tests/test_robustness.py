import csv

import numpy as np
import pytest

from signs.analysis.metrics import expected_calibration_error
from signs.analysis.robustness import (
    ROBUSTNESS_FILE,
    benchmark_run,
    conditions,
    degradation_table,
    summarize,
)
from signs.cli import main
from signs.config import load_config
from signs.data.degradations import SEVERITIES
from signs.training.trainer import train


def test_ece_is_zero_when_confidence_matches_accuracy():
    # 10 predictions at 0.8, 8 of them right.
    confidences = np.full(10, 0.8)
    correct = np.array([1] * 8 + [0] * 2)
    assert expected_calibration_error(confidences, correct) == pytest.approx(0)


def test_ece_measures_overconfidence():
    # Sure of every answer, right on half: ECE is 0.5.
    assert expected_calibration_error(np.ones(4), np.array([1, 0, 1, 0])) == pytest.approx(0.5)
    # Two bins: 0.95 right twice (gap 0.05), 0.55 wrong twice (gap 0.55).
    confidences = np.array([0.95, 0.95, 0.55, 0.55])
    correct = np.array([1, 1, 0, 0])
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.3)


def test_ece_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [1, 0])
    with pytest.raises(ValueError):
        expected_calibration_error([], [])


def test_conditions_start_clean_and_check_names():
    pairs = conditions()
    assert pairs[0] == ("clean", 0)
    assert len(pairs) == 1 + 8 * SEVERITIES
    assert conditions(["jpeg"], [2, 5]) == [("clean", 0), ("jpeg", 2), ("jpeg", 5)]
    with pytest.raises(ValueError, match="unknown degradation"):
        conditions(["rain"])
    with pytest.raises(ValueError, match="severities"):
        conditions(["jpeg"], [6])


def _read(run_dir):
    with (run_dir / ROBUSTNESS_FILE).open() as handle:
        return list(csv.DictReader(handle))


def test_benchmark_scores_a_network_with_calibration(tiny_config):
    tiny_config.train.epochs = 1
    train(tiny_config)
    rows = benchmark_run(tiny_config.run_dir, ["jpeg", "haze"], [1, 5])
    assert [(r["degradation"], r["severity"]) for r in rows] == [
        ("clean", 0),
        ("jpeg", 1),
        ("jpeg", 5),
        ("haze", 1),
        ("haze", 5),
    ]
    saved = _read(tiny_config.run_dir)
    assert len(saved) == 5
    for row in saved:
        assert 0 <= float(row["accuracy"]) <= 1
        assert 0 <= float(row["ece"]) <= 1
        assert int(row["images"]) == 43
    assert saved[1]["level"] == "40"


def test_benchmark_repeats_exactly(tiny_config):
    tiny_config.train.epochs = 1
    train(tiny_config)
    first = benchmark_run(tiny_config.run_dir, ["sensor_noise"], [5], seed=3)
    again = benchmark_run(tiny_config.run_dir, ["sensor_noise"], [5], seed=3, workers=2)
    assert first == again


def test_benchmark_scores_hog_without_calibration(fake_data_dir, tmp_path):
    config = load_config(
        None,
        [
            f"data.data_dir={fake_data_dir}",
            "data.split=random",
            "data.image_size=24",
            "model.name=hog_svm",
            "model.pretrained=false",
            "model.freeze_features=false",
            "model.hog.svm_c=0.1",
            f"runs_dir={tmp_path}",
            "run_name=hog",
        ],
    )
    train(config)
    rows = benchmark_run(config.run_dir, ["low_resolution"], [3])
    assert len(rows) == 2
    assert all(row["ece"] == "" for row in _read(config.run_dir))

    # A summary of HOG alone has accuracy figures but no ECE figure.
    out = tmp_path / "robustness"
    main(["robustness", f"hog={config.run_dir}", "--out", str(out)])
    assert (out / "robustness_accuracy.png").is_file()
    assert not (out / "robustness_ece.png").exists()


def test_summarize_averages_seeds_and_draws(tiny_config, tmp_path):
    tiny_config.train.epochs = 1
    run_dirs = []
    for seed in (0, 1):
        tiny_config.seed = seed
        tiny_config.run_name = f"tiny-seed{seed}"
        train(tiny_config)
        main(
            [
                "benchmark",
                str(tiny_config.run_dir),
                "--degradations",
                "jpeg",
                "--severities",
                "1",
                "2",
            ]
        )
        run_dirs.append(tiny_config.run_dir)

    summary = summarize({"tiny": run_dirs})
    assert [row["seeds"] for row in summary] == [2, 2, 2]
    accuracies = [float(r["accuracy"]) for d in run_dirs for r in _read(d)[:1]]
    assert summary[0]["accuracy_mean"] == pytest.approx(np.mean(accuracies), abs=1e-6)
    assert "| tiny |" in degradation_table(summary)

    out = tmp_path / "robustness"
    label = "tiny=" + ",".join(str(d) for d in run_dirs)
    main(["robustness", label, "--out", str(out)])
    for name in ("robustness.csv", "robustness.md", "robustness_heatmap.png"):
        assert (out / name).is_file()
    assert (out / "robustness_accuracy.png").is_file() and (out / "robustness_ece.png").is_file()


def test_summarize_rejects_runs_scored_on_different_conditions(tiny_config):
    tiny_config.train.epochs = 1
    run_dirs = []
    for seed, severities in ((0, [1]), (1, [2])):
        tiny_config.seed = seed
        tiny_config.run_name = f"tiny-seed{seed}"
        train(tiny_config)
        benchmark_run(tiny_config.run_dir, ["jpeg"], severities)
        run_dirs.append(tiny_config.run_dir)
    with pytest.raises(ValueError, match="same conditions"):
        summarize({"tiny": run_dirs})
    with pytest.raises(FileNotFoundError):
        summarize({"none": [run_dirs[0].parent / "missing"]})

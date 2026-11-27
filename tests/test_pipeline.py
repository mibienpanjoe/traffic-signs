"""Train, evaluate and predict end to end on the fake dataset."""

import csv
import io
import json

import pyarrow.parquet as pq
import pytest
from PIL import Image

from signs.class_names import CLASS_NAMES
from signs.cli import main
from signs.evaluate import evaluate
from signs.predict import predict
from signs.training.trainer import train


def _history(run_dir):
    with (run_dir / "history.csv").open() as handle:
        return list(csv.DictReader(handle))


def test_train_eval_predict_write_the_run_folder(tiny_config, fake_data_dir, tmp_path):
    summary = train(tiny_config)
    run_dir = tiny_config.run_dir
    for name in ["config.yaml", "history.csv", "model.pt", "train_summary.json"]:
        assert (run_dir / name).is_file(), name
    assert len(_history(run_dir)) == tiny_config.train.epochs
    assert summary["best_epoch"] in (1, 2)

    metrics = evaluate(run_dir)
    saved = json.loads((run_dir / "metrics.json").read_text())
    assert saved == metrics
    assert 0.0 <= saved["accuracy"] <= 1.0
    assert set(saved) >= {"accuracy", "macro_f1", "macro_precision", "macro_recall"}
    with (run_dir / "per_class.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == len(CLASS_NAMES)
    assert (run_dir / "confusion_matrix.png").is_file()

    raw = pq.read_table(fake_data_dir / "test.parquet").column("image")[0].as_py()["bytes"]
    image_path = tmp_path / "sign.png"
    Image.open(io.BytesIO(raw)).save(image_path)
    top = predict(run_dir, image_path, top_k=3)
    assert len(top) == 3
    assert all(name in CLASS_NAMES for name, _ in top)
    scores = [score for _, score in top]
    assert scores == sorted(scores, reverse=True)


def test_same_seed_gives_the_same_history(tiny_config):
    tiny_config.model.freeze_features = False
    tiny_config.run_name = "first"
    train(tiny_config)
    tiny_config.run_name = "second"
    train(tiny_config)
    runs = tiny_config.run_dir.parent

    def numbers(name):
        return [{k: v for k, v in row.items() if k != "seconds"} for row in _history(runs / name)]

    assert numbers("first") == numbers("second")


def test_finished_run_is_not_overwritten(tiny_config):
    train(tiny_config)
    with pytest.raises(FileExistsError):
        train(tiny_config)


def test_overwrite_removes_the_old_test_scores(tiny_config):
    train(tiny_config)
    evaluate(tiny_config.run_dir)
    assert (tiny_config.run_dir / "metrics.json").is_file()
    train(tiny_config, overwrite=True)
    assert not (tiny_config.run_dir / "metrics.json").exists()
    assert (tiny_config.run_dir / "model.pt").is_file()


def test_cli_trains_and_evaluates(tiny_config):
    data_dir = tiny_config.data.data_dir
    runs_dir = tiny_config.runs_dir
    main(
        [
            "train",
            "--set", f"data.data_dir={data_dir}",
            "--set", "data.image_size=32",
            "--set", "model.pretrained=false",
            "--set", "train.epochs=1",
            "--set", f"runs_dir={runs_dir}",
            "--run-name", "from-cli",
            "--seed", "3",
        ]
    )  # fmt: skip
    run_dir = tiny_config.run_dir.parent / "from-cli"
    assert json.loads((run_dir / "train_summary.json").read_text())["seed"] == 3
    main(["eval", "--run", str(run_dir)])
    assert (run_dir / "metrics.json").is_file()


def test_cli_reports_a_missing_run_without_a_traceback(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["eval", "--run", str(tmp_path / "missing")])
    assert stopped.value.code == 2
    assert "Train a run first" in capsys.readouterr().err


def test_top_k_outside_the_classes_is_rejected(tiny_config, tmp_path):
    train(tiny_config)
    with pytest.raises(ValueError, match="--top-k must be between 1 and 43"):
        predict(tiny_config.run_dir, tmp_path / "unused.png", top_k=50)

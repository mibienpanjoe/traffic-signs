import json

from signs.analysis.profile import profile_run
from signs.experiment import run_experiment, write_table
from signs.training.trainer import train


def test_profile_reports_size_and_latency(tiny_config):
    train(tiny_config)
    stats = profile_run(tiny_config.run_dir, images=8)
    assert stats["parameters"] > 1_000_000
    assert stats["file_megabytes"] > 1
    assert stats["latency_ms"] > 0
    assert stats["image_size"] == 32
    assert stats["latency_images"] == 8


def test_model_table_joins_scores_profile_and_training_time(tiny_config, tmp_path):
    tiny_config.train.epochs = 1
    results = tmp_path / "results"
    run_experiment(tiny_config, [0], results)
    run_dir = tiny_config.run_dir.parent / "tiny-seed0"
    (run_dir / "profile.json").write_text(
        json.dumps({"run_name": "tiny-seed0", "image_size": 32, "parameters": 1234,
                    "file_megabytes": 0.5, "latency_ms": 3.25})
    )  # fmt: skip
    table = write_table(
        [results / "tiny.json"], results / "models", kind="models", runs_dir=run_dir.parent
    )
    assert "| tiny | 32×32 | 1 |" in table
    assert "| 1,234 | 0.50 | 3.2 |" in table
    assert "profile_latency_ms" in (results / "models.csv").read_text()

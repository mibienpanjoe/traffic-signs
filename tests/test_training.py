import csv
import json

import pytest

from signs.config import TrainConfig, load_config
from signs.models.registry import build_model
from signs.training.trainer import parameter_groups, train


def _history(run_dir):
    with (run_dir / "history.csv").open() as handle:
        return list(csv.DictReader(handle))


def test_frozen_trunk_gives_one_group():
    config = load_config(None, ["model.pretrained=false"])
    groups = parameter_groups(build_model(config.model), config.train)
    assert [group["name"] for group in groups] == ["head"]


def test_trained_trunk_gets_its_own_learning_rate():
    config = load_config(
        None,
        ["model.pretrained=false", "model.freeze_features=false", "train.backbone_lr_scale=0.1"],
    )
    network = build_model(config.model)
    groups = {group["name"]: group for group in parameter_groups(network, config.train)}
    assert groups["trunk"]["lr"] == pytest.approx(config.train.lr * 0.1)
    assert groups["head"]["lr"] == pytest.approx(config.train.lr)
    total = sum(p.numel() for group in groups.values() for p in group["params"])
    assert total == sum(p.numel() for p in network.parameters())


def test_cosine_schedule_lowers_the_learning_rate(tiny_config):
    tiny_config.train.schedule = "cosine"
    tiny_config.train.epochs = 3
    train(tiny_config)
    lrs = [float(row["lr"]) for row in _history(tiny_config.run_dir)]
    assert lrs[0] == pytest.approx(tiny_config.train.lr)
    assert lrs[0] > lrs[1] > lrs[2] > 0
    assert (tiny_config.run_dir / "training_curves.png").is_file()


def test_early_stopping_ends_after_patience_epochs_without_gain(tiny_config, monkeypatch):
    import signs.training.trainer as trainer

    scores = iter([0.5, 0.4, 0.4, 0.9, 0.9])

    def fake_epoch(network, loader, criterion, device, optimizer=None):
        return (1.0, 0.5) if optimizer is not None else (1.0, next(scores))

    monkeypatch.setattr(trainer, "run_epoch", fake_epoch)
    tiny_config.train.epochs = 5
    tiny_config.train.early_stopping_patience = 2
    summary = train(tiny_config)
    assert summary["best_epoch"] == 1
    assert summary["epochs_run"] == 3
    assert summary["stopped_early"] is True
    saved = json.loads((tiny_config.run_dir / "train_summary.json").read_text())
    assert saved["epochs_run"] == 3


def test_bad_schedule_is_rejected():
    with pytest.raises(ValueError, match="train.schedule must be constant or cosine"):
        TrainConfig(schedule="step")

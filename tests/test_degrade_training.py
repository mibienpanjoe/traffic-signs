import numpy as np
import pytest
from PIL import Image

import signs.data.degradations as degradations
from signs.config import load_config
from signs.data.transforms import RandomDegrade, train_transform
from signs.training.trainer import train


@pytest.fixture
def picture():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8))


def test_config_defaults_leave_training_unchanged():
    augment = load_config(None).data.augment
    assert augment.degrade_probability == 0
    assert augment.degrade_names == []
    assert not any(
        isinstance(step, RandomDegrade)
        for step in train_transform(load_config(None).data).transforms
    )


@pytest.mark.parametrize(
    "override, message",
    [
        ("data.augment.degrade_probability=1.5", "between 0 and 1"),
        ("data.augment.degrade_names=[rain]", "unknown degradation"),
        ("data.augment.degrade_max_severity=6", "1 to 5"),
    ],
)
def test_config_rejects_bad_values(override, message):
    with pytest.raises(ValueError, match=message):
        load_config(None, [override])


def test_one_name_becomes_a_list():
    config = load_config(None, ["data.augment.degrade_names=jpeg"])
    assert config.data.augment.degrade_names == ["jpeg"]


def test_degrade_configs_hold_out_their_family():
    names = {
        run: load_config(f"configs/{run}.yaml").data.augment.degrade_names
        for run in ("cnn_degrade_no_blur", "cnn_degrade_no_noise", "cnn_degrade_no_haze")
    }
    assert not {"motion_blur", "defocus_blur"} & set(names["cnn_degrade_no_blur"])
    assert not {"sensor_noise", "low_light"} & set(names["cnn_degrade_no_noise"])
    assert "haze" not in names["cnn_degrade_no_haze"]
    for held_out in names.values():
        assert len(held_out) >= 6
    assert load_config("configs/cnn_degrade.yaml").data.augment.degrade_probability == 0.5


def test_random_degrade_follows_its_probability(picture):
    np.random.seed(0)
    never = RandomDegrade([], 0.0, 5)
    assert all(never(picture) is picture for _ in range(20))
    always = RandomDegrade(["sensor_noise"], 1.0, 5)
    assert not np.array_equal(np.asarray(always(picture)), np.asarray(picture))


def test_random_degrade_uses_only_its_names_and_severities(picture, monkeypatch):
    seen = []

    def record(image, name, severity, seed=0):
        seen.append((name, severity))
        return image

    monkeypatch.setattr(degradations, "degrade", record)
    np.random.seed(1)
    step = RandomDegrade(["jpeg", "haze"], 1.0, 2)
    for _ in range(200):
        step(picture)
    assert {name for name, _ in seen} == {"jpeg", "haze"}
    assert {severity for _, severity in seen} == {1, 2}


def test_random_degrade_repeats_with_the_same_seed(picture):
    step = RandomDegrade([], 0.7, 5)
    np.random.seed(3)
    first = [np.asarray(step(picture)) for _ in range(10)]
    np.random.seed(3)
    again = [np.asarray(step(picture)) for _ in range(10)]
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))


def test_training_on_degraded_images_runs_and_repeats(tiny_config, tmp_path):
    tiny_config.train.epochs = 1
    tiny_config.data.augment.degrade_probability = 1.0
    first = train(tiny_config)
    tiny_config.run_name = "tiny-again"
    again = train(tiny_config)
    assert first["best_val_accuracy"] == again["best_val_accuracy"]
    assert first["best_val_loss"] == again["best_val_loss"]

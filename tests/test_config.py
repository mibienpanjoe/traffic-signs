from pathlib import Path

import pytest

from signs.config import Config, load_config, save_config

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_match_base_yaml():
    from_file = load_config(ROOT / "configs" / "base.yaml")
    defaults = Config()
    defaults.run_name = from_file.run_name
    assert from_file == defaults


def test_experiment_file_inherits_from_base():
    config = load_config(ROOT / "configs" / "mobilenet_frozen.yaml")
    assert config.run_name == "mobilenet_frozen"
    assert config.data.image_size == 64
    assert config.model.freeze_features is True


def test_overrides_are_typed_and_nested():
    config = load_config(None, ["train.epochs=12", "train.lr=3e-4", "data.augment.contrast=0"])
    assert config.train.epochs == 12
    assert config.train.lr == pytest.approx(3e-4)
    assert config.data.augment.contrast == 0
    assert config.data.augment.brightness == 0.3


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(None, ["train.epoch=3"])


def test_base_merge_keeps_sibling_keys(tmp_path):
    (tmp_path / "base.yaml").write_text("data:\n  image_size: 96\n  batch_size: 16\n")
    (tmp_path / "child.yaml").write_text("base: base.yaml\ndata:\n  image_size: 128\n")
    config = load_config(tmp_path / "child.yaml")
    assert config.data.image_size == 128
    assert config.data.batch_size == 16


def test_saved_config_round_trips(tmp_path):
    config = load_config(None, ["seed=7", "model.freeze_features=false"])
    save_config(config, tmp_path / "config.yaml")
    assert load_config(tmp_path / "config.yaml") == config


def test_wrong_type_is_rejected():
    with pytest.raises(ValueError, match="train.epochs expects int"):
        load_config(None, ["train.epochs=five"])


@pytest.mark.parametrize("name", ["2024", "001", "yes", "null"])
def test_text_override_stays_text(name):
    assert load_config(None, [f"run_name={name}"]).run_name == name


def test_scalar_for_a_section_is_rejected():
    with pytest.raises(ValueError, match="data expects a mapping"):
        load_config(None, ["data=5"])


def test_zero_epochs_is_rejected():
    with pytest.raises(ValueError, match="train.epochs must be at least 1"):
        load_config(None, ["train.epochs=0"])


def test_base_cycle_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("base: b.yaml\n")
    (tmp_path / "b.yaml").write_text("base: a.yaml\n")
    with pytest.raises(ValueError, match="name each other as base"):
        load_config(tmp_path / "a.yaml")

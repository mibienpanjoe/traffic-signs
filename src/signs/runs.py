"""Everything a run leaves behind lives in one folder: runs/<run_name>/.

config.yaml          the full resolved config the run used
history.csv          one row per epoch
model.pt             weights with the best validation accuracy
train_summary.json   best epoch and its validation scores
training_curves.png  loss, accuracy and learning rate per epoch
metrics.json         validation and test scores, written by `signs eval`
per_class.csv        test precision, recall and F1 per class
confused_pairs.csv   the most frequent test mistakes
"""

import csv
import json
from pathlib import Path

import torch

from signs.config import Config, from_dict, save_config

CONFIG_FILE = "config.yaml"
CHECKPOINT_FILE = "model.pt"
# Every file train and eval write. --overwrite removes these and nothing else.
RUN_FILES = (
    CONFIG_FILE,
    CHECKPOINT_FILE,
    "history.csv",
    "train_summary.json",
    "training_curves.png",
    "metrics.json",
    "per_class.csv",
    "confused_pairs.csv",
    "classification_report.txt",
    "confusion_matrix.png",
)


def create_run_dir(config: Config, overwrite: bool = False) -> Path:
    run_dir = config.run_dir
    if (run_dir / CHECKPOINT_FILE).exists() and not overwrite:
        raise FileExistsError(
            f"{run_dir} already holds a trained model. Pick another --run-name or pass --overwrite."
        )
    if overwrite:
        # Old test scores or weights must not sit next to the new run's config.
        for name in RUN_FILES:
            (run_dir / name).unlink(missing_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / CONFIG_FILE)
    return run_dir


def load_run_config(run_dir: Path) -> Config:
    import yaml

    path = Path(run_dir) / CONFIG_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Train a run first: signs train")
    return from_dict(yaml.safe_load(path.read_text()))


def write_json(path: Path, values: dict) -> None:
    path.write_text(json.dumps(values, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write rows as CSV. With fieldnames, an empty list still writes the header."""
    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_trained_model(run_dir: Path, device: torch.device):
    """Rebuild the network a run trained and load its best weights.

    Returns (network in eval mode, the run's config).
    """
    from signs.models.registry import build_model

    run_dir = Path(run_dir)
    config = load_run_config(run_dir)
    # The checkpoint already holds every weight, so skip the ImageNet download.
    config.model.pretrained = False
    network = build_model(config.model).to(device)
    saved = torch.load(run_dir / CHECKPOINT_FILE, map_location=device, weights_only=True)
    network.load_state_dict(saved["state_dict"])
    network.eval()
    return network, config


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

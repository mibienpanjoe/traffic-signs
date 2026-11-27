"""Load GTSRB and turn sign photos into tensors the network can train on.

The usual torchvision download talks to sid.erda.dk. That host did not accept
a connection from this machine, so the images come from the public Hugging
Face copy of the same benchmark (tanganke/gtsrb): image bytes plus a class id.
"""

import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from signs.config import DataConfig
from signs.data.splits import random_split_indices
from signs.data.transforms import eval_transform, train_transform

_TABLES = {}


def _table(data_dir: Path, split: str):
    path = Path(data_dir) / f"{split}.parquet"
    key = str(path.resolve())
    if key not in _TABLES:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run: signs download")
        _TABLES[key] = pq.read_table(path)
    return _TABLES[key]


class SignDataset(Dataset):
    """One split of GTSRB. Each item is a PIL image (then transformed) and an int label."""

    def __init__(self, data_dir, split: str, transform, indices=None):
        table = _table(data_dir, split)
        self._bytes = table.column("image")
        self.labels = table.column("label").to_pylist()
        self.transform = transform
        self.indices = list(range(len(self.labels))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        raw = self._bytes[index].as_py()["bytes"]
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_splits(config: DataConfig):
    """Return train, validation, and test datasets.

    Validation is a share of the training images, held out so we can notice
    overfitting before we touch the test set.
    """
    n_train = _table(config.data_dir, "train").num_rows
    train_idx, val_idx = random_split_indices(n_train, config.val_fraction, config.split_seed)
    train_set = SignDataset(config.data_dir, "train", train_transform(config), train_idx)
    val_set = SignDataset(config.data_dir, "train", eval_transform(config), val_idx)
    test_set = SignDataset(config.data_dir, "test", eval_transform(config))
    return train_set, val_set, test_set


def load_test_set(config: DataConfig):
    return SignDataset(config.data_dir, "test", eval_transform(config))


def preview_dataset(config: DataConfig):
    """Training images before resize and normalization, for plotting."""
    return SignDataset(config.data_dir, "train", transform=None)


def make_loader(dataset, config: DataConfig, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

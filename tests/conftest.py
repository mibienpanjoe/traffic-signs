"""A tiny fake GTSRB, written in the same Parquet layout as the real tables.

Every class gets a few random 24×24 images, so the whole pipeline runs in
seconds on a CPU without downloading anything.
"""

import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from signs.class_names import CLASS_NAMES
from signs.config import load_config


def _table(per_class: int, rng: np.random.Generator) -> pa.Table:
    images, labels = [], []
    for class_id in range(len(CLASS_NAMES)):
        for _ in range(per_class):
            pixels = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            images.append({"bytes": buffer.getvalue(), "path": None})
            labels.append(class_id)
    return pa.table({"image": images, "label": labels})


@pytest.fixture(scope="session")
def fake_data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("hf")
    rng = np.random.default_rng(0)
    pq.write_table(_table(3, rng), data_dir / "train.parquet")
    pq.write_table(_table(1, rng), data_dir / "test.parquet")
    return data_dir


@pytest.fixture(scope="session")
def fake_track_dir(tmp_path_factory):
    """Training rows in track order: 2 tracks of 30 frames per class."""
    data_dir = tmp_path_factory.mktemp("hf_tracks")
    rng = np.random.default_rng(1)
    pq.write_table(_table(60, rng), data_dir / "train.parquet")
    pq.write_table(_table(1, rng), data_dir / "test.parquet")
    return data_dir


@pytest.fixture
def tiny_config(fake_data_dir, tmp_path):
    """A config that trains a random-init MobileNetV2 on the fake data, fast."""
    return load_config(
        None,
        [
            f"data.data_dir={fake_data_dir}",
            "data.image_size=32",
            "data.batch_size=32",
            # 3 frames per class do not form 30-frame tracks.
            "data.split=random",
            "model.pretrained=false",
            "train.epochs=2",
            f"runs_dir={tmp_path / 'runs'}",
            "run_name=tiny",
        ],
    )

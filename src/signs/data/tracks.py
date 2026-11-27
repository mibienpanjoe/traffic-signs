"""Check that the training table holds 30-frame tracks, in order.

The Hugging Face copy does not say where one track ends. Three signs in
the pixels show it:

- every block of 30 rows has a single class,
- consecutive frames inside a block look alike, and frames across a block
  boundary do not,
- inside a block the sign grows as the car drives towards it, then the
  size drops back at the next block.

The check also tries other block offsets: only offset 0 puts the
boundaries where the pictures change.
"""

import io

import numpy as np
from PIL import Image

from signs.config import DataConfig
from signs.data.dataset import _table
from signs.data.splits import TRACK_LENGTH


def _thumbnail(raw: bytes) -> tuple[np.ndarray, int]:
    image = Image.open(io.BytesIO(raw))
    width = image.size[0]
    gray = np.asarray(image.convert("L").resize((16, 16)), dtype=np.float64).ravel()
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    return gray / np.sqrt(gray.size), width


def track_structure(config: DataConfig, track_length: int = TRACK_LENGTH) -> dict:
    """Measure how well the training rows fit tracks of track_length frames."""
    table = _table(config.data_dir, "train")
    labels = np.array(table.column("label").to_pylist())
    n = len(labels)
    images = table.column("image")
    thumbs, widths = zip(*(_thumbnail(images[i].as_py()["bytes"]) for i in range(n)), strict=True)
    thumbs = np.stack(thumbs)
    widths = np.array(widths)

    # Correlation of each frame with the next one, in [-1, 1].
    similarity = (thumbs[:-1] * thumbs[1:]).sum(axis=1)
    grows = widths[1:] >= widths[:-1]

    def at_boundary(offset):
        return (np.arange(n - 1) - offset) % track_length == track_length - 1

    boundary = at_boundary(0)
    whole_blocks = n % track_length == 0
    single_class = bool(
        whole_blocks and (labels.reshape(-1, track_length) == labels[::track_length, None]).all()
    )
    return {
        "rows": int(n),
        "track_length": track_length,
        "tracks": int(n // track_length),
        "rows_divide_into_tracks": bool(whole_blocks),
        "one_class_per_track": single_class,
        "similarity_within_track": round(float(similarity[~boundary].mean()), 3),
        "similarity_across_boundary": round(float(similarity[boundary].mean()), 3),
        "size_grows_within_track": round(float(grows[~boundary].mean()), 3),
        "size_grows_across_boundary": round(float(grows[boundary].mean()), 3),
        "boundary_similarity_by_offset": {
            offset: round(float(similarity[at_boundary(offset)].mean()), 3)
            for offset in range(0, track_length, 5)
        },
    }

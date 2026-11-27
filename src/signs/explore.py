"""Look at the dataset before any training: counts, sizes, and a sample grid."""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from signs.class_names import CLASS_NAMES  # noqa: E402
from signs.config import DataConfig  # noqa: E402
from signs.data.dataset import preview_dataset  # noqa: E402


def explore(config: DataConfig, output: Path) -> Path:
    """Print class counts and save one photo per class. Returns the figure path."""
    dataset = preview_dataset(config)
    labels = dataset.labels
    counts = Counter(labels)

    print(f"training images: {len(dataset)}")
    print(f"classes: {len(CLASS_NAMES)}")
    print()
    print(f"{'id':>3}  {'count':>6}  name")
    for class_id, name in enumerate(CLASS_NAMES):
        print(f"{class_id:>3}  {counts[class_id]:>6}  {name}")

    widest = max(counts.values())
    rarest = min(counts.values())
    print()
    print(f"most common class has {widest} images, rarest has {rarest}")

    first_index = {}
    for index, label in enumerate(labels):
        first_index.setdefault(label, index)
        if len(first_index) == len(CLASS_NAMES):
            break

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(6, 8, figsize=(12, 9))
    for axis, class_id in zip(axes.ravel(), range(len(CLASS_NAMES)), strict=False):
        image, label = dataset[first_index[class_id]]
        axis.imshow(image)
        axis.set_title(str(label), fontsize=8)
        axis.axis("off")
    for axis in axes.ravel()[43:]:
        axis.axis("off")
    figure.suptitle("One training photo per class (title is the class id)")
    figure.tight_layout()
    figure_path = output / "sample_signs.png"
    figure.savefig(figure_path, dpi=120)
    plt.close(figure)
    print(f"saved {figure_path}")
    return figure_path

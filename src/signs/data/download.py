"""Download the GTSRB tables this project trains on.

The files are the public Hugging Face copy of the German Traffic Sign
Recognition Benchmark (tanganke/gtsrb). They stay in data/hf/, which git
ignores, so the repository itself stays small.
"""

import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/tanganke/gtsrb/resolve/main/data"
FILES = {
    "train.parquet": f"{BASE}/train-00000-of-00001.parquet",
    "test.parquet": f"{BASE}/test-00000-of-00001.parquet",
}


def download(dest: Path) -> None:
    """Fetch each table into dest, skipping a file that is already there."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        path = dest / name
        if path.is_file() and path.stat().st_size > 1_000_000:
            print(f"already present: {path}")
            continue
        print(f"downloading {name}")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(path)
        print(f"saved {path} ({path.stat().st_size} bytes)")

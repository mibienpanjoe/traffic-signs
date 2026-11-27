"""Classify one sign photo with a trained run."""

from pathlib import Path

import torch
from PIL import Image

from signs.class_names import CLASS_NAMES
from signs.data.transforms import eval_transform
from signs.runs import load_trained_model, pick_device


def predict(run_dir: Path, image_path: Path, top_k: int = 3) -> list[tuple[str, float]]:
    """Return the top_k (class name, softmax score) pairs for one image."""
    if not 1 <= top_k <= len(CLASS_NAMES):
        raise ValueError(f"--top-k must be between 1 and {len(CLASS_NAMES)}, got {top_k}")
    device = pick_device()
    network, config = load_trained_model(run_dir, device)
    image = Image.open(image_path).convert("RGB")
    batch = eval_transform(config.data)(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(network(batch), dim=1)[0]

    top = torch.topk(probabilities, k=top_k)
    return [
        (CLASS_NAMES[index], score)
        for score, index in zip(top.values.tolist(), top.indices.tolist(), strict=True)
    ]

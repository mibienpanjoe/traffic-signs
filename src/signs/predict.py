"""Classify one sign photo with a trained run."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from signs.class_names import CLASS_NAMES
from signs.data.transforms import eval_transform
from signs.runs import SVM_FILE, load_run_config, load_trained_model, pick_device


def predict(run_dir: Path, image_path: Path, top_k: int = 3) -> list[tuple[str, float]]:
    """Return the top_k (class name, score) pairs for one image, best first.

    For a network the score is the softmax probability. For HOG + SVM it is
    the SVM's signed distance to each class's hyperplane, not a probability.
    """
    if not 1 <= top_k <= len(CLASS_NAMES):
        raise ValueError(f"--top-k must be between 1 and {len(CLASS_NAMES)}, got {top_k}")
    image = Image.open(image_path).convert("RGB")
    if load_run_config(run_dir).model.name == "hog_svm":
        return _predict_hog(run_dir, image, top_k)
    device = pick_device()
    network, config = load_trained_model(run_dir, device)
    batch = eval_transform(config.data)(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(network(batch), dim=1)[0]

    top = torch.topk(probabilities, k=top_k)
    return [
        (CLASS_NAMES[index], score)
        for score, index in zip(top.values.tolist(), top.indices.tolist(), strict=True)
    ]


def _predict_hog(run_dir: Path, image: Image.Image, top_k: int) -> list[tuple[str, float]]:
    from signs.models.hog_svm import HogSVM

    model = HogSVM.load(Path(run_dir) / SVM_FILE)
    scores = model.decision(model.transform(image)[None, :])[0]
    order = np.argsort(scores)[::-1][:top_k]
    classes = model.svm.classes_
    return [(CLASS_NAMES[int(classes[i])], float(scores[i])) for i in order]

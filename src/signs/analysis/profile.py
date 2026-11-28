"""How big and how fast a trained model is, on this CPU.

Latency is measured end to end for one image, the way `signs predict`
uses a model: from a decoded photo through resizing (and HOG for the
SVM) to class scores. Each image is timed alone (batch of 1), after a
few warm-up calls, and the median is kept because the slowest calls
are mostly the operating system, not the model.
"""

import os
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from signs.data.dataset import load_test_set
from signs.runs import CHECKPOINT_FILE, SVM_FILE, load_run_config, load_trained_model


def _test_images(config, count: int) -> list[Image.Image]:
    dataset = load_test_set(config.data)
    dataset.transform = None
    step = max(1, len(dataset) // count)
    return [dataset[i][0] for i in range(0, step * count, step)]


def _time(call, images, warmup: int = 5) -> float:
    for image in images[:warmup]:
        call(image)
    seconds = []
    for image in images:
        started = time.perf_counter()
        call(image)
        seconds.append(time.perf_counter() - started)
    return statistics.median(seconds)


def profile_run(run_dir: Path, images: int = 200, threads: int = 1) -> dict:
    """Parameters, file size and median single-image CPU latency of a run's model."""
    run_dir = Path(run_dir)
    config = load_run_config(run_dir)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        samples = _test_images(config, images)
        if config.model.name == "hog_svm":
            from signs.models.hog_svm import HogSVM

            model = HogSVM.load(run_dir / SVM_FILE)
            parameters = int(model.svm.coef_.size + model.svm.intercept_.size)
            path = run_dir / SVM_FILE

            def call(image):
                return model.decision(model.transform(image)[None, :])

        else:
            from signs.data.transforms import eval_transform

            network, config = load_trained_model(run_dir, torch.device("cpu"))
            transform = eval_transform(config.data)
            parameters = sum(p.numel() for p in network.parameters())
            path = run_dir / CHECKPOINT_FILE

            def call(image):
                with torch.inference_mode():
                    return network(transform(image).unsqueeze(0))

        latency = _time(call, samples)
    finally:
        torch.set_num_threads(previous_threads)

    return {
        "run_name": config.run_name,
        "model": config.model.name,
        "image_size": config.data.image_size,
        "parameters": parameters,
        "file_megabytes": round(os.path.getsize(path) / 1e6, 3),
        "latency_ms": round(1000 * latency, 3),
        "latency_threads": threads,
        "latency_images": len(samples),
    }

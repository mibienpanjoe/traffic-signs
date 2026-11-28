"""Fit HOG + linear SVM and keep the regularization strength that validates best."""

import time

import numpy as np
from torch.utils.data import DataLoader

from signs.config import Config
from signs.data.dataset import load_splits
from signs.models.hog_svm import HogSVM, HogTransform, fit_svm
from signs.runs import SVM_FILE, create_run_dir, write_csv, write_json
from signs.training.seed import seed_everything

FEATURE_WORKERS = 4


def extract_features(dataset, workers: int = FEATURE_WORKERS):
    """Run a HOG dataset once and stack (features, labels) as NumPy arrays."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=workers)
    features, labels = [], []
    for batch, batch_labels in loader:
        features.append(batch.numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(features), np.concatenate(labels)


def hog_transform(config: Config) -> HogTransform:
    return HogTransform(config.data.image_size, config.model.hog)


def train_hog_svm(config: Config, overwrite: bool = False) -> dict:
    """Fit one SVM per value of model.hog.svm_c, keep the best on validation.

    There is no augmentation: HOG features are computed once per image.
    history.csv has one row per C instead of one per epoch.
    """
    run_dir = create_run_dir(config, overwrite=overwrite)
    seed_everything(config.seed)
    transform = hog_transform(config)
    print(f"run: {run_dir}   HOG features per image: {transform.feature_size()}")

    train_set, val_set, _ = load_splits(config.data, train_tf=transform, eval_tf=transform)
    started = time.perf_counter()
    train_x, train_y = extract_features(train_set)
    val_x, val_y = extract_features(val_set)
    feature_seconds = time.perf_counter() - started
    print(f"train images: {len(train_y)}   val images: {len(val_y)}   ({feature_seconds:.0f}s)")

    history = []
    best = None
    for c in config.model.hog.svm_c:
        started = time.perf_counter()
        svm = fit_svm(train_x, train_y, c=c, seed=config.seed)
        seconds = time.perf_counter() - started
        train_acc = float(np.mean(svm.predict(train_x) == train_y))
        val_acc = float(np.mean(svm.predict(val_x) == val_y))
        history.append(
            {
                "svm_c": c,
                "train_accuracy": round(train_acc, 6),
                "val_accuracy": round(val_acc, 6),
                "seconds": round(seconds, 2),
            }
        )
        write_csv(run_dir / "history.csv", history)
        print(f"C {c:g}  train acc {train_acc:.3f}  val acc {val_acc:.3f}  ({seconds:.0f}s)")
        if best is None or val_acc > best["val_accuracy"]:
            best = {"svm_c": c, "val_accuracy": val_acc, "svm": svm}

    HogSVM(transform, best["svm"]).save(run_dir / SVM_FILE)
    summary = {
        "run_name": config.run_name,
        "seed": config.seed,
        "device": "cpu",
        "train_images": len(train_y),
        "val_images": len(val_y),
        "best_svm_c": best["svm_c"],
        "best_val_accuracy": best["val_accuracy"],
        "features_per_image": transform.feature_size(),
        "trainable_parameters": int(best["svm"].coef_.size + best["svm"].intercept_.size),
        "train_seconds": round(feature_seconds + sum(row["seconds"] for row in history), 1),
    }
    write_json(run_dir / "train_summary.json", summary)
    print(f"best validation accuracy: {best['val_accuracy']:.3f} (C {best['svm_c']:g})")
    return summary

"""Score trained runs on the test set under every degradation and severity.

`benchmark_run` writes runs/<run>/robustness.csv: one row for the clean
test set, then one per degradation × severity. Each row has accuracy,
macro-F1, and for networks the expected calibration error (ECE) and mean
confidence. HOG + SVM gives scores, not probabilities, so its ECE is
left empty.

`summarize` collects several models, each over one or more seeds, into
mean ± std per model × degradation × severity.
"""

import copy
import csv
from pathlib import Path

import numpy as np
import torch

from signs.analysis.aggregate import mean_std
from signs.analysis.metrics import classification_scores, expected_calibration_error
from signs.data.dataset import SignDataset, make_loader
from signs.data.degradations import DEGRADATIONS, SEVERITIES, DegradedDataset
from signs.data.transforms import eval_transform
from signs.runs import SVM_FILE, load_run_config, load_trained_model, pick_device, write_csv

ROBUSTNESS_FILE = "robustness.csv"
# The same benchmark with each degraded image restored first.
RESTORED_FILE = "robustness_restored.csv"
FIELDS = [
    "run_name",
    "model",
    "degradation",
    "severity",
    "level",
    "images",
    "accuracy",
    "macro_f1",
    "ece",
    "mean_confidence",
]
CLEAN = "clean"


def conditions(names=None, severities=None) -> list[tuple[str, int]]:
    """(degradation, severity) pairs to score, the clean test set first."""
    names = list(DEGRADATIONS) if names is None else list(names)
    severities = list(range(1, SEVERITIES + 1)) if severities is None else list(severities)
    for name in names:
        if name not in DEGRADATIONS:
            raise ValueError(f"unknown degradation {name!r}; known: {', '.join(DEGRADATIONS)}")
    for severity in severities:
        if not 1 <= severity <= SEVERITIES:
            raise ValueError(f"severities go from 1 to {SEVERITIES}, got {severity}")
    return [(CLEAN, 0)] + [(name, severity) for name in names for severity in severities]


class _Scorer:
    """Turns a dataset of test images into (truths, predictions, confidences or None)."""

    def __init__(self, run_dir: Path, workers: int | None):
        self.config = load_run_config(run_dir)
        self.data = copy.deepcopy(self.config.data)
        if workers is not None:
            self.data.num_workers = workers
        self.hog = self.config.model.name == "hog_svm"
        if self.hog:
            from signs.models.hog_svm import HogSVM

            self.model = HogSVM.load(Path(run_dir) / SVM_FILE)
            self.transform = self.model.transform
        else:
            self.device = pick_device()
            self.model, _ = load_trained_model(run_dir, self.device)
            self.transform = eval_transform(self.config.data)

    def __call__(self, dataset):
        if self.hog:
            from signs.training.hog_trainer import extract_features

            features, truths = extract_features(dataset, self.data)
            return truths, self.model.predict(features), None
        truths, predictions, confidences = [], [], []
        with torch.no_grad():
            for images, labels in make_loader(dataset, self.data, shuffle=False):
                probabilities = torch.softmax(self.model(images.to(self.device)), dim=1)
                confidence, predicted = probabilities.max(dim=1)
                truths.extend(labels.tolist())
                predictions.extend(predicted.cpu().tolist())
                confidences.extend(confidence.cpu().tolist())
        return np.array(truths), np.array(predictions), np.array(confidences)


def benchmark_run(
    run_dir: Path,
    names=None,
    severities=None,
    seed: int = 0,
    workers: int | None = None,
    restore: bool = False,
) -> list[dict]:
    """Score one trained run on the clean and degraded test set.

    Writes robustness.csv, or with `restore` robustness_restored.csv, where
    each degraded image goes through its classical restoration first.
    """
    run_dir = Path(run_dir)
    scorer = _Scorer(run_dir, workers)
    rows = []
    for name, severity in conditions(names, severities):
        data_dir = scorer.config.data.data_dir
        if name == CLEAN:
            dataset, level = SignDataset(data_dir, "test", transform=scorer.transform), ""
        else:
            raw = SignDataset(data_dir, "test", transform=None)
            dataset = DegradedDataset(raw, name, severity, scorer.transform, seed, restore)
            level = DEGRADATIONS[name].levels[severity - 1]
        truths, predictions, confidences = scorer(dataset)
        scores = classification_scores(truths, predictions)
        row = {
            "run_name": scorer.config.run_name,
            "model": scorer.config.model.name,
            "degradation": name,
            "severity": severity,
            "level": level,
            "images": len(truths),
            "accuracy": round(scores["accuracy"], 6),
            "macro_f1": round(scores["macro_f1"], 6),
            "ece": "",
            "mean_confidence": "",
        }
        if confidences is not None:
            correct = predictions == truths
            row["ece"] = round(expected_calibration_error(confidences, correct), 6)
            row["mean_confidence"] = round(float(confidences.mean()), 6)
        rows.append(row)
        ece = f"  ECE {row['ece']:.3f}" if row["ece"] != "" else ""
        print(f"{run_dir.name}  {name:>14} {severity}  accuracy {row['accuracy']:.3f}{ece}")
    write_csv(run_dir / (RESTORED_FILE if restore else ROBUSTNESS_FILE), rows, fieldnames=FIELDS)
    return rows


def _read(source: Path) -> list[dict]:
    """A benchmark table: a run folder means its robustness.csv, a file is read as given."""
    source = Path(source)
    path = source if source.suffix == ".csv" else source / ROBUSTNESS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run: signs benchmark {path.parent}")
    with path.open() as handle:
        return list(csv.DictReader(handle))


def summarize(models: dict[str, list[Path]]) -> list[dict]:
    """Mean ± std over seeds, per model label × degradation × severity.

    models maps a label to the run folders of its seeds (their
    robustness.csv), or to benchmark CSV files such as
    runs/<run>/robustness_restored.csv. Every run of a label must have
    scored the same conditions.
    """
    summary = []
    for label, run_dirs in models.items():
        if not run_dirs:
            raise ValueError(f"no runs given for {label}")
        tables = [_read(run_dir) for run_dir in run_dirs]
        keys = [[(r["degradation"], r["severity"]) for r in table] for table in tables]
        if any(k != keys[0] for k in keys):
            raise ValueError(f"the runs of {label} did not score the same conditions")
        for position, (name, severity) in enumerate(keys[0]):
            rows = [table[position] for table in tables]
            out = {
                "model": label,
                "degradation": name,
                "severity": int(severity),
                "level": rows[0]["level"],
                "seeds": len(rows),
            }
            for metric in ("accuracy", "macro_f1", "ece", "mean_confidence"):
                values = [row[metric] for row in rows]
                if "" in values:
                    out[f"{metric}_mean"] = out[f"{metric}_std"] = ""
                    continue
                mean, std = mean_std([float(v) for v in values])
                out[f"{metric}_mean"] = round(mean, 6)
                out[f"{metric}_std"] = round(std, 6)
            summary.append(out)
    return summary


def degradation_table(summary: list[dict]) -> str:
    """Markdown: clean accuracy and mean accuracy over severities 1–5, per degradation."""
    models = list(dict.fromkeys(row["model"] for row in summary))
    names = [n for n in dict.fromkeys(row["degradation"] for row in summary) if n != CLEAN]
    titles = [DEGRADATIONS[n].title for n in names]
    lines = [
        "| Model | Clean | " + " | ".join(titles) + " | Mean degraded |",
        "|---" * (len(names) + 3) + "|",
    ]
    for model in models:
        rows = [row for row in summary if row["model"] == model]
        clean = next(row["accuracy_mean"] for row in rows if row["degradation"] == CLEAN)
        means = [
            np.mean([row["accuracy_mean"] for row in rows if row["degradation"] == name])
            for name in names
        ]
        cells = [f"{100 * clean:.1f}"] + [f"{100 * m:.1f}" for m in means]
        lines.append(f"| {model} | " + " | ".join(cells) + f" | {100 * np.mean(means):.1f} |")
    return "\n".join(lines) + "\n"

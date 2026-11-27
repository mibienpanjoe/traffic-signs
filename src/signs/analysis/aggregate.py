"""Mean and spread of a configuration's scores over several seeds.

One run is one draw: another seed gives other weights and other numbers.
A difference between two configurations means something only when it is
larger than that seed-to-seed spread, so results are reported as
mean ± sample standard deviation over seeds.
"""

import json
import statistics
from pathlib import Path

METRICS = ("accuracy", "macro_f1", "macro_precision", "macro_recall")


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def aggregate_runs(run_dirs: list[Path], name: str) -> dict:
    """Combine the metrics.json of several runs of one configuration."""
    if not run_dirs:
        raise ValueError(f"no runs to aggregate for {name}")
    loaded = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Run: signs eval --run {run_dir}")
        loaded.append(json.loads(path.read_text()))

    splits = {metrics["split"] for metrics in loaded}
    if len(splits) > 1:
        raise ValueError(f"runs of {name} use different splits: {sorted(splits)}")
    seeds = [metrics["seed"] for metrics in loaded]
    if len(set(seeds)) < len(seeds):
        raise ValueError(f"runs of {name} repeat a seed: {seeds}")

    row = {"name": name, "split": splits.pop(), "seeds": len(loaded)}
    for part in ("val", "test"):
        for metric in METRICS:
            mean, std = mean_std([metrics[part][metric] for metrics in loaded])
            row[f"{part}_{metric}_mean"] = round(mean, 6)
            row[f"{part}_{metric}_std"] = round(std, 6)
    gaps = [metrics["val"]["accuracy"] - metrics["test"]["accuracy"] for metrics in loaded]
    mean, std = mean_std(gaps)
    row["val_minus_test_accuracy_mean"] = round(mean, 6)
    row["val_minus_test_accuracy_std"] = round(std, 6)
    row["runs"] = " ".join(str(Path(run_dir).name) for run_dir in run_dirs)
    return row


def markdown_table(rows: list[dict]) -> str:
    """A results table for the README: mean ± std over seeds, in percent."""

    def cell(row, key):
        return f"{100 * row[key + '_mean']:.1f} ± {100 * row[key + '_std']:.1f}"

    lines = [
        "| Configuration | Split | Seeds | Val accuracy | Test accuracy | Test macro-F1 "
        "| Val − test |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['split']} | {row['seeds']} "
            f"| {cell(row, 'val_accuracy')} | {cell(row, 'test_accuracy')} "
            f"| {cell(row, 'test_macro_f1')} | {cell(row, 'val_minus_test_accuracy')} |"
        )
    return "\n".join(lines) + "\n"

"""Train and evaluate one configuration over several seeds, then summarize.

Each seed is an ordinary run in runs/<run_name>-seed<N>/. A seed whose run
is already trained and evaluated is skipped, so an interrupted experiment
picks up where it stopped. A finished seed trained with a different
config is an error, not a silent reuse. The summary goes to <results_dir>/<run_name>.json.
"""

import copy
import json
from pathlib import Path

from signs.analysis.aggregate import aggregate_runs, markdown_table, model_table
from signs.config import Config
from signs.runs import has_model, load_run_config, write_csv, write_json


def seed_run(config: Config, seed: int) -> Config:
    run = copy.deepcopy(config)
    run.seed = seed
    run.run_name = f"{config.run_name}-seed{seed}"
    return run


def same_training(saved: Config, wanted: Config) -> bool:
    """Whether a saved run was trained the way `wanted` asks.

    Where the files live (data_dir, runs_dir) does not change the weights,
    so it is left out. num_workers stays in: each worker draws its own
    augmentations, so the worker count changes the numbers.
    """
    saved, wanted = copy.deepcopy(saved), copy.deepcopy(wanted)
    for config in (saved, wanted):
        config.runs_dir = ""
        config.data.data_dir = ""
    return saved == wanted


def run_experiment(
    config: Config, seeds: list[int], results_dir: Path, overwrite: bool = False
) -> dict:
    from signs.evaluate import evaluate
    from signs.training.trainer import train

    if len(set(seeds)) < len(seeds):
        raise ValueError(f"seeds repeat: {seeds}")
    run_dirs = []
    for seed in seeds:
        run = seed_run(config, seed)
        # train_summary.json is written only after the last epoch, so a run
        # stopped halfway has a checkpoint but does not count as trained.
        trained = has_model(run.run_dir) and (run.run_dir / "train_summary.json").is_file()
        evaluated = (run.run_dir / "metrics.json").is_file()
        if trained and not overwrite:
            if not same_training(load_run_config(run.run_dir), run):
                raise ValueError(
                    f"{run.run_dir} was trained with another config. "
                    "Pass --overwrite to retrain it, or pick another run_name."
                )
            if evaluated:
                print(f"{run.run_dir}: already trained and evaluated, skipping")
            else:
                print(f"{run.run_dir}: already trained, evaluating")
                evaluate(run.run_dir)
        else:
            train(run, overwrite=True)
            evaluate(run.run_dir)
        run_dirs.append(run.run_dir)

    summary = aggregate_runs(run_dirs, config.run_name)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_dir / f"{config.run_name}.json", summary)
    print(markdown_table([summary]))
    print(f"saved {results_dir / (config.run_name + '.json')}")
    return summary


def table_paths(out_stem: Path) -> tuple[Path, Path]:
    """<out_stem>.csv and <out_stem>.md, keeping any dot already in the name."""
    out_stem = Path(out_stem)
    return (
        out_stem.parent / f"{out_stem.name}.csv",
        out_stem.parent / f"{out_stem.name}.md",
    )


def _with_run_details(row: dict, runs_dir: Path) -> dict:
    """Add the first run's profile and the mean training time over the seeds."""
    names = row.get("runs", "").split()
    row = dict(row)
    profile_path = Path(runs_dir) / names[0] / "profile.json" if names else None
    if profile_path and profile_path.is_file():
        for key, value in json.loads(profile_path.read_text()).items():
            if key != "run_name":
                row[f"profile_{key}"] = value
    seconds = []
    for name in names:
        summary_path = Path(runs_dir) / name / "train_summary.json"
        if summary_path.is_file():
            seconds.append(json.loads(summary_path.read_text()).get("train_seconds"))
    if seconds and None not in seconds:
        row["train_minutes"] = round(sum(seconds) / len(seconds) / 60, 2)
    return row


def write_table(
    summary_paths: list[Path], out_stem: Path, kind: str = "split", runs_dir: Path = Path("runs")
) -> str:
    """Collect experiment summaries into <out_stem>.csv and <out_stem>.md.

    kind "split" compares validation with test (Phase 2). kind "models"
    compares test scores with size, latency and training time.
    """
    rows = [json.loads(Path(path).read_text()) for path in summary_paths]
    if not rows:
        raise ValueError("no experiment summaries given")
    if kind not in ("split", "models"):
        raise ValueError(f"unknown table kind {kind!r}; known: split, models")
    if kind == "models":
        rows = [_with_run_details(row, runs_dir) for row in rows]
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(table_paths(out_stem)[0], rows, fieldnames=fieldnames)
    table = markdown_table(rows) if kind == "split" else model_table(rows)
    table_paths(out_stem)[1].write_text(table)
    return table

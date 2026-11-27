"""Train and evaluate one configuration over several seeds, then summarize.

Each seed is an ordinary run in runs/<run_name>-seed<N>/. A seed whose run
is already trained and evaluated is skipped, so an interrupted experiment
picks up where it stopped. The summary goes to <results_dir>/<run_name>.json.
"""

import copy
import json
from pathlib import Path

from signs.analysis.aggregate import aggregate_runs, markdown_table
from signs.config import Config
from signs.runs import CHECKPOINT_FILE, write_csv, write_json


def seed_run(config: Config, seed: int) -> Config:
    run = copy.deepcopy(config)
    run.seed = seed
    run.run_name = f"{config.run_name}-seed{seed}"
    return run


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
        done = (run.run_dir / CHECKPOINT_FILE).is_file() and (
            run.run_dir / "metrics.json"
        ).is_file()
        if done and not overwrite:
            print(f"{run.run_dir}: already trained and evaluated, skipping")
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


def write_table(summary_paths: list[Path], out_stem: Path) -> str:
    """Collect experiment summaries into <out_stem>.csv and <out_stem>.md."""
    rows = [json.loads(Path(path).read_text()) for path in summary_paths]
    if not rows:
        raise ValueError("no experiment summaries given")
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_stem.with_suffix(".csv"), rows)
    table = markdown_table(rows)
    out_stem.with_suffix(".md").write_text(table)
    return table

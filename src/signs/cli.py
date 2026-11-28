"""The `signs` command: one entry point for every step of the pipeline.

signs download
signs explore
signs train   --config configs/mobilenet_frozen.yaml [--set train.epochs=10]
signs eval    --run runs/mobilenet_frozen
signs predict path/to/sign.jpg --run runs/mobilenet_frozen
"""

import argparse
from pathlib import Path

from signs.config import load_config

DEFAULT_CONFIG = "configs/mobilenet_frozen.yaml"


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"YAML config (default: {DEFAULT_CONFIG} when present, else built-in defaults)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one config value, e.g. --set train.epochs=10 (repeatable)",
    )


def _config(args):
    path = args.config
    if path is None and Path(DEFAULT_CONFIG).is_file():
        path = Path(DEFAULT_CONFIG)
    overrides = list(args.overrides)
    if getattr(args, "run_name", None):
        overrides.append(f"run_name={args.run_name}")
    if getattr(args, "seed", None) is not None:
        overrides.append(f"seed={args.seed}")
    return load_config(path, overrides)


def _default_run(args) -> Path:
    if args.run is not None:
        return args.run
    config = load_config(Path(DEFAULT_CONFIG) if Path(DEFAULT_CONFIG).is_file() else None)
    return config.run_dir


def cmd_download(args) -> None:
    from signs.data.download import download

    download(Path(_config(args).data.data_dir))


def cmd_explore(args) -> None:
    from signs.explore import explore

    explore(_config(args).data, args.out)


def cmd_tracks(args) -> None:
    from signs.data.tracks import track_structure
    from signs.runs import write_json

    stats = track_structure(_config(args).data)
    for key, value in stats.items():
        print(f"{key}: {value}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, stats)
        print(f"saved {args.out}")


def cmd_hog_figure(args) -> None:
    from signs.data.dataset import preview_dataset
    from signs.models.hog_svm import HogTransform
    from signs.viz.figures import plot_hog_examples

    config = _config(args)
    transform = HogTransform(config.data.image_size, config.model.hog)
    plot_hog_examples(preview_dataset(config.data), transform, args.classes, args.out)
    print(f"saved {args.out}")


def cmd_train(args) -> None:
    from signs.training.trainer import train

    train(_config(args), overwrite=args.overwrite)


def cmd_experiment(args) -> None:
    from signs.experiment import run_experiment

    run_experiment(_config(args), args.seeds, args.results, overwrite=args.overwrite)


def cmd_table(args) -> None:
    from signs.experiment import table_paths, write_table

    print(write_table(args.summaries, args.out, kind=args.kind, runs_dir=args.runs_dir), end="")
    csv_path, md_path = table_paths(args.out)
    print(f"saved {csv_path} and {md_path}")


def cmd_profile(args) -> None:
    from signs.analysis.profile import profile_run
    from signs.runs import write_json

    for run in args.runs:
        stats = profile_run(run, images=args.images, threads=args.threads)
        write_json(Path(run) / "profile.json", stats)
        print(
            f"{run}: {stats['parameters']:,} parameters, {stats['file_megabytes']:.2f} MB, "
            f"{stats['latency_ms']:.1f} ms/image on {stats['latency_threads']} thread(s)"
        )


def cmd_eval(args) -> None:
    from signs.evaluate import evaluate

    evaluate(_default_run(args))


def cmd_predict(args) -> None:
    from signs.predict import predict
    from signs.runs import load_run_config

    run = _default_run(args)
    top = predict(run, args.image, top_k=args.top_k)
    svm = load_run_config(run).model.name == "hog_svm"
    print(f"file: {args.image}" + ("   (SVM scores, not probabilities)" if svm else ""))
    for name, score in top:
        print(f"  {score:+7.3f}  {name}" if svm else f"  {score:6.1%}  {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signs", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="fetch the GTSRB tables into data/hf/")
    _add_config_args(download)
    download.set_defaults(func=cmd_download)

    explore = commands.add_parser("explore", help="class counts and one photo per class")
    _add_config_args(explore)
    explore.add_argument("--out", type=Path, default=Path("runs/explore"))
    explore.set_defaults(func=cmd_explore)

    tracks = commands.add_parser("tracks", help="check that training rows form 30-frame tracks")
    _add_config_args(tracks)
    tracks.add_argument("--out", type=Path, default=None, help="also save the numbers as JSON")
    tracks.set_defaults(func=cmd_tracks)

    hog_figure = commands.add_parser("hog-figure", help="draw what HOG sees on a few signs")
    _add_config_args(hog_figure)
    hog_figure.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[1, 14, 13, 17, 25, 38],
        help="class ids to show (default: 30 km/h, stop, yield, no entry, road work, keep right)",
    )
    hog_figure.add_argument("--out", type=Path, default=Path("results/figures/hog_examples.png"))
    hog_figure.set_defaults(func=cmd_hog_figure)

    train = commands.add_parser("train", help="train a run into runs/<run_name>/")
    _add_config_args(train)
    train.add_argument("--run-name", help="override run_name from the config")
    train.add_argument("--seed", type=int, help="override the training seed")
    train.add_argument("--overwrite", action="store_true", help="replace an existing run")
    train.set_defaults(func=cmd_train)

    experiment = commands.add_parser(
        "experiment", help="train and evaluate one config over several seeds"
    )
    _add_config_args(experiment)
    experiment.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    experiment.add_argument("--results", type=Path, default=Path("results"))
    experiment.add_argument("--overwrite", action="store_true", help="retrain finished seeds")
    experiment.set_defaults(func=cmd_experiment)

    table = commands.add_parser("table", help="collect experiment summaries into one table")
    table.add_argument("summaries", type=Path, nargs="+", help="results/<name>.json files")
    table.add_argument("--out", type=Path, default=Path("results/table"))
    table.add_argument(
        "--kind",
        choices=["split", "models"],
        default="split",
        help="split: validation vs test; models: test scores with size, latency, training time",
    )
    table.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="where the seed runs are (models)"
    )
    table.set_defaults(func=cmd_table)

    profile = commands.add_parser(
        "profile", help="parameters, file size and CPU latency of trained runs"
    )
    profile.add_argument("runs", type=Path, nargs="+", help="run folders")
    profile.add_argument("--images", type=int, default=200, help="test images to time")
    profile.add_argument("--threads", type=int, default=1, help="CPU threads for PyTorch")
    profile.set_defaults(func=cmd_profile)

    run_help = "run folder (default: the run of the default config)"
    evaluate = commands.add_parser("eval", help="score a run on the official test set")
    evaluate.add_argument("--run", type=Path, default=None, help=run_help)
    evaluate.set_defaults(func=cmd_eval)

    predict = commands.add_parser("predict", help="classify one image")
    predict.add_argument("image", type=Path)
    predict.add_argument("--run", type=Path, default=None, help=run_help)
    predict.add_argument("--top-k", type=int, default=3)
    predict.set_defaults(func=cmd_predict)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.exit(2, f"signs {args.command}: {error}\n")


if __name__ == "__main__":
    main()

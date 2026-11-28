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


def cmd_degrade(args) -> None:
    from PIL import Image

    from signs.class_names import CLASS_NAMES
    from signs.data.dataset import SignDataset
    from signs.viz.figures import plot_degradation_grid

    if args.image is not None:
        image, title = Image.open(args.image).convert("RGB"), ""
    else:
        test = SignDataset(_config(args).data.data_dir, "test", transform=None)
        for index, label in enumerate(test.labels):
            image, _ = test[index]
            if label == args.class_id and min(image.size) >= args.min_size:
                break
        else:
            raise ValueError(
                f"no test image of class {args.class_id} is at least {args.min_size} pixels across"
            )
        title = (
            f"{CLASS_NAMES[args.class_id]}, test image {index} ({image.size[0]}×{image.size[1]}): "
            "each degradation at severities 1 to 5"
        )
    if args.restore:
        from signs.viz.figures import plot_restoration_examples

        out = args.out or Path("results/figures/restorations.png")
        plot_restoration_examples(image, out, seed=args.seed)
    else:
        out = args.out or Path("results/figures/degradations.png")
        plot_degradation_grid(image, out, seed=args.seed, title=title)
    print(f"saved {out}")


def cmd_restoration_quality(args) -> None:
    from signs.analysis.restoration_quality import restoration_quality
    from signs.runs import write_csv

    rows = restoration_quality(_config(args).data.data_dir, images=args.images, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)
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


def cmd_curves(args) -> None:
    from signs.viz.figures import plot_validation_curves

    runs = {}
    for text in args.runs:
        label, _, folder = text.rpartition("=")
        runs[label or Path(folder).name] = Path(folder)
    plot_validation_curves(runs, args.out)
    print(f"saved {args.out}")


def cmd_benchmark(args) -> None:
    from signs.analysis.robustness import RESTORED_FILE, ROBUSTNESS_FILE, benchmark_run

    name = RESTORED_FILE if args.restore else ROBUSTNESS_FILE
    for run in args.runs:
        if (Path(run) / name).is_file() and not args.overwrite:
            print(f"{run}: {name} exists, skipping (--overwrite to score again)")
            continue
        benchmark_run(
            run,
            args.degradations,
            args.severities,
            seed=args.seed,
            workers=args.workers,
            restore=args.restore,
        )
        print(f"saved {Path(run) / name}")


def cmd_robustness(args) -> None:
    from signs.analysis.robustness import degradation_table, summarize
    from signs.runs import write_csv
    from signs.viz.figures import plot_robustness_curves, plot_robustness_heatmap

    models = {}
    for text in args.models:
        label, _, folders = text.rpartition("=")
        paths = [Path(folder) for folder in folders.split(",") if folder]
        if not label:
            first = paths[0]
            restored = first.name == "robustness_restored.csv"
            label = (
                (first.parent.name + " restored")
                if restored
                else (first.parent.name if first.suffix == ".csv" else first.name)
            )
        if label in models:
            raise ValueError(f"two models are labelled {label!r}; name them with LABEL=")
        models[label] = paths
    summary = summarize(models)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "robustness.csv", summary)
    table = degradation_table(summary)
    (out / "robustness.md").write_text(table)
    plot_robustness_heatmap(summary, out / "robustness_heatmap.png")
    plot_robustness_curves(summary, out / "robustness_accuracy.png", metric="accuracy")
    figures = 2
    # HOG + SVM has no probabilities: without a network there is no ECE to draw.
    if any(row["ece_mean"] != "" for row in summary):
        plot_robustness_curves(summary, out / "robustness_ece.png", metric="ece")
        figures += 1
    print(table, end="")
    print(f"saved robustness.csv, robustness.md and {figures} figures in {out}")


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

    degrade = commands.add_parser("degrade", help="draw one sign under every degradation")
    _add_config_args(degrade)
    degrade.add_argument("--image", type=Path, default=None, help="a photo (default: a test sign)")
    degrade.add_argument("--class-id", type=int, default=1, help="class of the test sign")
    degrade.add_argument(
        "--min-size", type=int, default=64, help="smallest side of the test sign, in pixels"
    )
    degrade.add_argument("--seed", type=int, default=0, help="seed for noise and blur direction")
    degrade.add_argument(
        "--restore",
        action="store_true",
        help="draw each degradation next to its restoration (results/figures/restorations.png)",
    )
    degrade.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: results/figures/degradations.png, or restorations.png with --restore",
    )
    degrade.set_defaults(func=cmd_degrade)

    quality = commands.add_parser(
        "restoration-quality", help="PSNR and SSIM of degraded and restored test images"
    )
    _add_config_args(quality)
    quality.add_argument("--images", type=int, default=500, help="test images, spread evenly")
    quality.add_argument("--seed", type=int, default=0, help="seed for noise and blur direction")
    quality.add_argument("--out", type=Path, default=Path("results/restoration_quality.csv"))
    quality.set_defaults(func=cmd_restoration_quality)

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

    curves = commands.add_parser("curves", help="validation curves of several runs, together")
    curves.add_argument(
        "runs", nargs="+", metavar="[LABEL=]RUN", help="run folders, optionally labelled"
    )
    curves.add_argument("--out", type=Path, default=Path("results/curves.png"))
    curves.set_defaults(func=cmd_curves)

    benchmark = commands.add_parser(
        "benchmark", help="score runs on the test set under every degradation and severity"
    )
    benchmark.add_argument("runs", type=Path, nargs="+", help="run folders")
    benchmark.add_argument(
        "--degradations", nargs="+", default=None, help="names to score (default: all)"
    )
    benchmark.add_argument(
        "--severities", type=int, nargs="+", default=None, help="severities (default: 1 to 5)"
    )
    benchmark.add_argument("--seed", type=int, default=0, help="seed for noise and blur direction")
    benchmark.add_argument(
        "--workers", type=int, default=None, help="loader workers (default: the run's config)"
    )
    benchmark.add_argument(
        "--restore",
        action="store_true",
        help="restore each degraded image first; writes robustness_restored.csv",
    )
    benchmark.add_argument("--overwrite", action="store_true", help="score runs again")
    benchmark.set_defaults(func=cmd_benchmark)

    robustness = commands.add_parser(
        "robustness", help="compare benchmarked models: table, heatmap and curves"
    )
    robustness.add_argument(
        "models",
        nargs="+",
        metavar="[LABEL=]RUN[,RUN...]",
        help=(
            "one model per argument; several runs (seeds) separated by commas. "
            "A run folder means its robustness.csv; a .csv path is read as given"
        ),
    )
    robustness.add_argument("--out", type=Path, default=Path("results/robustness"))
    robustness.set_defaults(func=cmd_robustness)

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

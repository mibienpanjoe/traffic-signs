# Traffic sign recognition

A camera sees one cropped traffic sign. This project names it, and reports how sure the model is. That label is what a small robot or a driver-assist demo can turn into an action: stop, yield, or a speed limit.

The model is a MobileNetV2 network. Its convolutional layers stay as they were trained on ImageNet. A new final layer learns the 43 sign classes of the German Traffic Sign Recognition Benchmark (GTSRB).

The study notes, with the libraries and the training concepts, are in [GUIDE.md](GUIDE.md).

## Layout

| Path | Role |
|---|---|
| `src/signs/cli.py` | The `signs` command: `download`, `explore`, `train`, `eval`, `predict` |
| `src/signs/config.py` | One config object, read from YAML, with `--set key=value` overrides |
| `src/signs/data/` | Download, load, split, augment, and batch the photos |
| `src/signs/models/` | The network, built from its config name |
| `src/signs/training/` | Training loop and seeding |
| `src/signs/evaluate.py` | Accuracy, macro-F1, per-class report, confusion matrix |
| `src/signs/predict.py` | Classify one image |
| `src/signs/runs.py` | Where a run writes its files, and how a trained run is loaded back |
| `src/signs/class_names.py` | The 43 names, in official id order |
| `configs/` | `base.yaml` defaults and one file per experiment |

`data/` and `runs/` are created on your machine and are listed in `.gitignore`. The dataset is a few hundred megabytes. Checkpoints, histories and plots are produced by the commands.

## Setup

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `uv.lock`, and PyTorch comes from the CPU wheel index.

```bash
uv sync
uv run signs download
```

`signs download` skips a file that is already downloaded. This copy has 26,640 training photos and 12,630 test photos, 43 classes.

Without uv, `python3 -m pip install -e .` installs the same package and the `signs` command, with unpinned versions.

## Run

```bash
uv run signs explore
uv run signs train --config configs/mobilenet_frozen.yaml
uv run signs eval --run runs/mobilenet_frozen
uv run signs predict path/to/a_sign.jpg --run runs/mobilenet_frozen
```

Each training run gets its own folder, `runs/<run_name>/`:

| File | Written by | Content |
|---|---|---|
| `config.yaml` | `train` | The full resolved config the run used |
| `history.csv` | `train` | Loss, accuracy, learning rate and time per epoch |
| `model.pt` | `train` | Weights with the best validation accuracy |
| `train_summary.json` | `train` | Best epoch and its validation scores |
| `metrics.json` | `eval` | Test accuracy, macro precision, recall and F1 |
| `per_class.csv` | `eval` | Test precision, recall and F1 for every class |
| `classification_report.txt`, `confusion_matrix.png` | `eval` | The same, for reading |

The test images are not used to choose the checkpoint. `eval` and `predict` rebuild the model from the run's own `config.yaml`.

Change one value without a new file, or repeat a run with another seed:

```bash
uv run signs train --set train.epochs=10 --run-name frozen-10ep
uv run signs train --seed 1 --run-name mobilenet_frozen-seed1
```

On a CPU, the same seed gives the same weights and the same numbers. On a GPU, repeats are close but not guaranteed to be exact. The validation split has its own `data.split_seed`, so changing the training seed does not change which images are held out. `train` refuses to replace a finished run unless you pass `--overwrite`.

There is no GPU requirement. Five epochs on a CPU is the intended first run.

## After the presentation

Natural next steps, without changing the problem:

- Train the convolutional layers as well, with a smaller learning rate.
- Compare this frozen MobileNetV2 with a small network trained from scratch.
- Show the per-class errors, especially speed limits, which share a shape and differ by the digits.

## License

Apache License 2.0. See [LICENSE](LICENSE).

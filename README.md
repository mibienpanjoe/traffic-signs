# Traffic sign recognition

[![CI](https://github.com/mibienpanjoe/traffic-signs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mibienpanjoe/traffic-signs/actions/workflows/ci.yml)

A camera sees one cropped traffic sign. This project names it, and reports how sure the model is. That label is what a small robot or a driver-assist demo can turn into an action: stop, yield, or a speed limit.

The model is a MobileNetV2 network. Its convolutional layers stay as they were trained on ImageNet. A new final layer learns the 43 sign classes of the German Traffic Sign Recognition Benchmark (GTSRB).

The study notes, with the libraries and the training concepts, are in [GUIDE.md](GUIDE.md).

## Results so far

Frozen MobileNetV2 (only the last layer trained), 64×64 input, 5 epochs on CPU, 3 seeds. Mean ± standard deviation over seeds, in percent:

| Configuration | Split | Seeds | Val accuracy | Test accuracy | Test macro-F1 | Val − test |
|---|---|---|---|---|---|---|
| mobilenet_frozen | track | 3 | 44.0 ± 0.7 | 45.5 ± 1.1 | 35.4 ± 1.1 | -1.5 ± 0.8 |
| mobilenet_frozen_random_split | random | 3 | 64.3 ± 0.6 | 46.3 ± 0.7 | 36.5 ± 0.8 | 17.9 ± 0.3 |

**Validation must hold out whole tracks.** Each physical sign is a track of 30 near-identical video frames. With the old random frame split, validation accuracy is 18 points above test accuracy. Validation then scores signs the model already trained on, from a frame further along the road. With whole tracks held out, validation is within 2 points of test. Test accuracy barely moves (45.5 vs 46.3), so the random split did not train a different model, it misreported it. Track structure evidence: [`results/track_structure.json`](results/track_structure.json).

**This baseline is weak, and accuracy hides how weak.** Test accuracy is 45.5%, but macro-F1 is 35.4%: many of the rare classes do much worse than the common ones. Pedestrians is never recognized (recall 0 in all 3 seeds), and End of no passing, Ice or snow and Curve right stay under 5%. Yield, No entry and Priority road, which have distinctive shapes, reach 80–85%.

**The mistakes are the expected ones.** The most frequent confusions, averaged over seeds, are Speed limit 120 → 80 (30% of 120 signs), 30 → 50 (29%), Ahead only → Keep right (24%) and 50 → 70 (18%). Speed limits share a red ring and differ only in their digits, which at 64×64 through a frozen ImageNet trunk is little detail.

Reproduce:

```bash
uv run signs tracks --out results/track_structure.json
uv run signs experiment --config configs/mobilenet_frozen.yaml --seeds 0 1 2
uv run signs experiment --config configs/mobilenet_frozen_random_split.yaml --seeds 0 1 2
uv run signs table results/mobilenet_frozen.json results/mobilenet_frozen_random_split.json \
    --out results/split_comparison
```

## Layout

| Path | Role |
|---|---|
| `src/signs/cli.py` | The `signs` command: `download`, `explore`, `train`, `eval`, `predict` |
| `src/signs/config.py` | One config object, read from YAML, with `--set key=value` overrides |
| `src/signs/data/` | Download, load, split, augment, and batch the photos |
| `src/signs/models/` | The networks (MobileNetV2, a small CNN), built from their config name |
| `src/signs/training/` | Training loop and seeding |
| `src/signs/evaluate.py` | Validation and test scores, per-class report, confused pairs, confusion matrix |
| `src/signs/analysis/metrics.py` | Accuracy, macro-F1, per-class scores, most-confused pairs |
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
| `train_summary.json` | `train` | Best epoch, its validation scores, epochs run, trainable parameters, training time |
| `training_curves.png` | `train` | Loss, accuracy and learning rate per epoch |
| `metrics.json` | `eval` | Accuracy and macro precision, recall and F1, on validation and on test |
| `per_class.csv` | `eval` | Test precision, recall, F1 and image count for every class |
| `confused_pairs.csv` | `eval` | The most frequent test mistakes, true class → predicted class, with names |
| `confusion_matrix.png` | `eval` | Test confusion matrix with class names, each row as shares of the true class |
| `classification_report.txt` | `eval` | The per-class test scores, for reading |

The test images are not used to choose the checkpoint. `eval` and `predict` rebuild the model from the run's own `config.yaml`.

Change one value without a new file, or repeat a run with another seed:

```bash
uv run signs train --set train.epochs=10 --run-name frozen-10ep
uv run signs train --seed 1 --run-name mobilenet_frozen-seed1
```

On a CPU, the same seed gives the same weights and the same numbers. On a GPU, repeats are close but not guaranteed to be exact. The validation split has its own `data.split_seed`, so changing the training seed does not change which images are held out. `train` refuses to replace a finished run unless you pass `--overwrite`.

There is no GPU requirement. Five epochs on a CPU is the intended first run.

### Several seeds

One run is one draw. Results are reported as mean ± standard deviation over seeds:

```bash
uv run signs experiment --config configs/mobilenet_frozen.yaml --seeds 0 1 2
uv run signs table results/mobilenet_frozen.json results/mobilenet_frozen_random_split.json \
    --out results/split_comparison
```

`experiment` trains and evaluates `runs/<run_name>-seed<N>/` for each seed, skips seeds that are already done, and writes `results/<run_name>.json`. `table` collects several of those into one CSV and Markdown table. `results/` is committed and holds only these small files.

## Development

```bash
uv sync
uv run pre-commit install   # ruff lint and format on every commit
uv run pytest               # runs on a tiny fake dataset, no download
```

CI runs the same lint and tests on pushes to `main` and on every pull request, on Python 3.10 and 3.12.

## After the presentation

Natural next steps, without changing the problem:

- Train the convolutional layers as well, with a smaller learning rate.
- Compare this frozen MobileNetV2 with a small network trained from scratch.
- Show the per-class errors, especially speed limits, which share a shape and differ by the digits.

## License

Apache License 2.0. See [LICENSE](LICENSE).

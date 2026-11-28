# Traffic sign recognition

[![CI](https://github.com/mibienpanjoe/traffic-signs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mibienpanjoe/traffic-signs/actions/workflows/ci.yml)

A camera sees one cropped traffic sign. This project names it, and reports how sure the model is. That label is what a small robot or a driver-assist demo can turn into an action: stop, yield, or a speed limit.

It compares four models on the 43 sign classes of the German Traffic Sign Recognition Benchmark (GTSRB): MobileNetV2 pretrained on ImageNet, frozen or fine-tuned; a small network trained from scratch; and hand-made HOG features with a linear SVM.

The study notes, with the libraries and the training concepts, are in [GUIDE.md](GUIDE.md).

## Results so far

### Validation split

Frozen MobileNetV2 (only the last layer trained), 64×64 input, 5 epochs on CPU, 3 seeds. Mean ± standard deviation over seeds, in percent:

| Configuration | Split | Seeds | Val accuracy | Test accuracy | Test macro-F1 | Val − test |
|---|---|---|---|---|---|---|
| mobilenet_frozen | track | 3 | 44.0 ± 0.7 | 45.5 ± 1.1 | 35.4 ± 1.1 | -1.5 ± 0.8 |
| mobilenet_frozen_random_split | random | 3 | 64.3 ± 0.6 | 46.3 ± 0.7 | 36.5 ± 0.8 | 17.9 ± 0.3 |

**Validation must hold out whole tracks.** Each physical sign is a track of 30 near-identical video frames. With the old random frame split, validation accuracy is 18 points above test accuracy. Validation then scores signs the model already trained on, from a frame further along the road. With whole tracks held out, validation is within 2 points of test. Test accuracy barely moves (45.5 vs 46.3), so the random split did not train a different model, it misreported it. Track structure evidence: [`results/track_structure.json`](results/track_structure.json).

**This baseline is weak, and accuracy hides how weak.** Test accuracy is 45.5%, but macro-F1 is 35.4%: many of the rare classes do much worse than the common ones. Pedestrians is never recognized (recall 0 in all 3 seeds), and End of no passing, Ice or snow and Curve right stay under 5%. Yield, No entry and Priority road, which have distinctive shapes, reach 80–85%.

**The mistakes are the expected ones.** The most frequent confusions, averaged over seeds, are Speed limit 120 → 80 (30% of 120 signs), 30 → 50 (29%), Ahead only → Keep right (24%) and 50 → 70 (18%). Speed limits share a red ring and differ only in their digits, which at 64×64 through a frozen ImageNet trunk is little detail.

### Comparing models

Four models at 64×64, all on the track split, scored on the official test set. Latency is the median time to classify one image, end to end, on one CPU thread.

| Model | Seeds | Test accuracy | Test macro-F1 | Parameters | File (MB) | CPU latency (ms/image) | Training (min) |
|---|---|---|---|---|---|---|---|
| MobileNetV2, frozen (cosine schedule) | 3 | 46.5 ± 0.3 | 36.6 ± 0.1 | 2,278,955 | 9.34 | 5.1 | 8 |
| MobileNetV2, fine-tuned | 3 | 86.6 ± 0.3 | 81.1 ± 0.9 | 2,278,955 | 9.34 | 4.8 | 23 |
| Small CNN from scratch | 3 | 95.3 ± 0.4 | 92.1 ± 1.1 | 75,099 | 0.31 | 2.0 | 20 |
| HOG + linear SVM | 1 | 91.2 | 90.1 | 75,895 | 0.61 | 1.0 | 5 |

**At 64×64, a small network trained from scratch wins.** It is 30 times smaller than MobileNetV2, more than twice as fast on a CPU, and 9 points more accurate than the fine-tuned MobileNetV2. Pedestrians, never recognized by the frozen model, is still the weakest class of the fine-tuned one (recall 15% on seed 0).

**Hand-made features are a strong baseline.** HOG, the histogram of edge directions in 8×8 cells, with a linear SVM reaches 91.2% in 5 minutes of CPU, with no network at all. What HOG sees: [`results/figures/hog_examples.png`](results/figures/hog_examples.png).

**Training the pretrained layers is what matters, not the schedule.** A cosine learning-rate schedule moves the frozen model from 45.5% to 46.5%. Training the convolutional layers as well, at a tenth of the learning rate, takes it to 86.6%. The validation curves: [`results/phase3/validation_curves.png`](results/phase3/validation_curves.png).

**MobileNetV2 wants larger images; the small CNN, with the same 15 epochs, does not get there.** Test accuracy in percent; 64×64 is the mean of 3 seeds, 96×96 and 128×128 are one seed each:

| Model | 64×64 | 96×96 | 128×128 |
|---|---|---|---|
| MobileNetV2, frozen | 46.5 | 58.7 | 64.7 |
| MobileNetV2, fine-tuned | 86.6 | 94.2 | 95.1 |
| Small CNN from scratch | 95.3 | 86.9 | 69.3 |

Fine-tuned at 128×128, MobileNetV2 matches the small CNN at 64×64 (95.1%), for four times the pixels, 30 times the parameters and four times the training time. The small CNN's drop at larger sizes is undertraining, not a limit of the size: at 96×96 it was still improving at epoch 15, with training accuracy at 87% against 96% at 64×64. Full table: [`results/phase3/input_size.md`](results/phase3/input_size.md).

**The hard classes are still the speed limits.** Most of the frequent mistakes are between speed limits (30 ↔ 50 and 80 → 50 for MobileNetV2 and HOG, 20 → 70 for the small CNN), which share a red ring and differ by their digits.

Reproduce:

```bash
bash scripts/phase3.sh   # about 8 hours on a laptop CPU
uv run signs tracks --out results/track_structure.json
uv run signs experiment --config configs/mobilenet_frozen.yaml --seeds 0 1 2
uv run signs experiment --config configs/mobilenet_frozen_random_split.yaml --seeds 0 1 2
uv run signs table results/mobilenet_frozen.json results/mobilenet_frozen_random_split.json \
    --out results/split_comparison
```

## Layout

| Path | Role |
|---|---|
| `src/signs/cli.py` | The `signs` command: `download`, `explore`, `tracks`, `train`, `eval`, `predict`, `experiment`, `table`, `hog-figure`, `profile`, `curves`, `degrade` |
| `src/signs/config.py` | One config object, read from YAML, with `--set key=value` overrides |
| `src/signs/data/` | Download, load, split, augment, degrade, and batch the photos |
| `src/signs/models/` | MobileNetV2, a small CNN, and HOG features with a linear SVM, built from their config name |
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

## Next steps

- Train the small CNN longer at 96×96 and 128×128, to see whether larger images help it once it converges.
- Measure how each model holds up when the image is blurred, darkened, noisy or compressed.

## License

Apache License 2.0. See [LICENSE](LICENSE).

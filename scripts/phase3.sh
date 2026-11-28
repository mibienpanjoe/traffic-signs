#!/usr/bin/env bash
# Reproduce every Phase 3 table: classical vs. deep baselines, and input size.
# On an 8-core CPU this takes several hours; finished seeds are skipped on a rerun.
set -euo pipefail

R=results/phase3

# Main comparison at 64x64, 3 seeds each.
uv run signs experiment --config configs/mobilenet_finetune.yaml --results $R
uv run signs experiment --config configs/mobilenet_frozen_cosine.yaml --results $R
uv run signs experiment --config configs/cnn_scratch.yaml --results $R
# The primal linear SVM is deterministic: other seeds give the same model.
uv run signs experiment --config configs/hog_svm.yaml --seeds 0 --results $R

# Input size, one seed each.
for size in 96 128; do
  for config in mobilenet_frozen_cosine mobilenet_finetune cnn_scratch; do
    uv run signs experiment --config configs/$config.yaml --seeds 0 --results $R \
      --set data.image_size=$size --set run_name=${config}_${size}
  done
done

# Size and latency, measured with nothing else running.
uv run signs profile runs/*-seed0

uv run signs table $R/mobilenet_frozen_cosine.json $R/mobilenet_finetune.json \
  $R/cnn_scratch.json $R/hog_svm.json --kind models --out $R/models_64
uv run signs table \
  $R/mobilenet_frozen_cosine.json $R/mobilenet_frozen_cosine_96.json $R/mobilenet_frozen_cosine_128.json \
  $R/mobilenet_finetune.json $R/mobilenet_finetune_96.json $R/mobilenet_finetune_128.json \
  $R/cnn_scratch.json $R/cnn_scratch_96.json $R/cnn_scratch_128.json \
  --kind models --out $R/input_size

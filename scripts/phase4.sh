#!/usr/bin/env bash
# Phase 4: score the Phase 3 models on the degraded test set.
# Needs the trained runs of scripts/phase3.sh. About 2 hours on a laptop CPU.
set -euo pipefail

R=results/phase4
WORKERS=${WORKERS:-4}

runs=(
    runs/mobilenet_frozen_cosine-seed{0,1,2}
    runs/mobilenet_finetune-seed{0,1,2}
    runs/cnn_scratch-seed{0,1,2}
    runs/hog_svm-seed0
    runs/mobilenet_finetune_128-seed0
)
uv run signs benchmark "${runs[@]}" --workers "$WORKERS"

uv run signs robustness \
    "MobileNetV2 frozen=runs/mobilenet_frozen_cosine-seed0,runs/mobilenet_frozen_cosine-seed1,runs/mobilenet_frozen_cosine-seed2" \
    "MobileNetV2 fine-tuned=runs/mobilenet_finetune-seed0,runs/mobilenet_finetune-seed1,runs/mobilenet_finetune-seed2" \
    "MobileNetV2 fine-tuned, 128 px=runs/mobilenet_finetune_128-seed0" \
    "small CNN=runs/cnn_scratch-seed0,runs/cnn_scratch-seed1,runs/cnn_scratch-seed2" \
    "HOG + SVM=runs/hog_svm-seed0" \
    --out "$R"

uv run signs degrade --out results/figures/degradations.png

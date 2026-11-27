"""Build a network from its config name."""

import torch.nn as nn

from signs.class_names import CLASS_NAMES
from signs.config import ModelConfig
from signs.models.mobilenet import build_mobilenet

MODELS = {
    "mobilenet_v2": build_mobilenet,
}


def build_model(config: ModelConfig, num_classes: int = len(CLASS_NAMES)) -> nn.Module:
    if config.name not in MODELS:
        raise ValueError(f"unknown model {config.name!r}; known: {sorted(MODELS)}")
    return MODELS[config.name](
        num_classes,
        pretrained=config.pretrained,
        freeze_features=config.freeze_features,
    )

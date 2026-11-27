import pytest
import torch

from signs.config import ModelConfig
from signs.models.registry import build_model


def test_output_has_one_logit_per_class():
    network = build_model(ModelConfig(pretrained=False)).eval()
    with torch.no_grad():
        logits = network(torch.zeros(2, 3, 64, 64))
    assert logits.shape == (2, 43)


def test_frozen_mode_trains_only_the_classifier():
    network = build_model(ModelConfig(pretrained=False, freeze_features=True))
    assert not any(p.requires_grad for p in network.features.parameters())
    assert all(p.requires_grad for p in network.classifier.parameters())


def test_unfrozen_mode_trains_everything():
    network = build_model(ModelConfig(pretrained=False, freeze_features=False))
    assert all(p.requires_grad for p in network.parameters())


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        build_model(ModelConfig(name="nope", pretrained=False))

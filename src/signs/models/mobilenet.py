"""A small pretrained vision network, adapted to the traffic-sign classes."""

import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_mobilenet(
    num_classes: int, pretrained: bool = True, freeze_features: bool = True
) -> nn.Module:
    """MobileNetV2 with a new final layer for GTSRB.

    The convolutional trunk already detects edges, colors, and simple shapes
    from ImageNet. We keep those weights and only replace the classifier,
    which originally predicted 1000 everyday object categories.
    `pretrained=False` starts from random weights, which tests use to avoid
    a download.
    """
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    network = mobilenet_v2(weights=weights)

    if freeze_features:
        for parameter in network.features.parameters():
            parameter.requires_grad = False

    in_features = network.classifier[1].in_features
    network.classifier[1] = nn.Linear(in_features, num_classes)
    return network

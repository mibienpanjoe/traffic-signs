"""A small convolutional network trained from random weights.

It answers one question: is ImageNet pretraining needed at all for
cropped, centred traffic signs? Three blocks of two 3×3 convolutions
(16, 32 and 64 channels), each block halving the image, then a global
average and one linear layer: 75k parameters, against 2.3M for
MobileNetV2. It works at any input size.
"""

import torch.nn as nn


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, width: int = 16, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, width),
            _block(width, 2 * width),
            _block(2 * width, 4 * width),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def build_small_cnn(
    num_classes: int, pretrained: bool = False, freeze_features: bool = False
) -> nn.Module:
    if pretrained or freeze_features:
        raise ValueError(
            "small_cnn has no pretrained weights: set model.pretrained=false "
            "and model.freeze_features=false"
        )
    return SmallCNN(num_classes)

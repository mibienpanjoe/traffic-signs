"""Turn a sign photo into the tensor the network expects."""

from torchvision import transforms

from signs.config import DataConfig

# MobileNetV2 was pretrained on ImageNet with these channel means and stds.
# Matching them keeps the pretrained features in the range they expect.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(config: DataConfig):
    """Augment training photos so the model sees more lighting and tilt.

    Signs are not mirrored: a flipped "keep right" is a different sign,
    so there is no horizontal flip here.
    """
    size = config.image_size
    augment = config.augment
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomRotation(augment.rotation_degrees),
            transforms.ColorJitter(
                brightness=augment.brightness,
                contrast=augment.contrast,
                saturation=augment.saturation,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(config: DataConfig):
    """Resize and normalize only. Evaluation must not randomly change the photo."""
    size = config.image_size
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

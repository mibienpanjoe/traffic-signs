"""Turn a sign photo into the tensor the network expects."""

import numpy as np
from torchvision import transforms

from signs.config import DataConfig

# MobileNetV2 was pretrained on ImageNet with these channel means and stds.
# Matching them keeps the pretrained features in the range they expect.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomDegrade:
    """With probability p, apply one random degradation at a random severity.

    Randomness comes from NumPy's global generator, which `seed_everything`
    seeds in the main process and `worker_init_fn` in each loader worker,
    so a run repeats. The degradation happens on the photo at its own
    size, as in the benchmark, before resizing.
    """

    def __init__(self, names, probability: float, max_severity: int):
        from signs.data.degradations import DEGRADATIONS

        self.names = list(names) or list(DEGRADATIONS)
        self.probability = probability
        self.max_severity = max_severity

    def __call__(self, image):
        from signs.data.degradations import degrade

        if np.random.random() >= self.probability:
            return image
        name = self.names[np.random.randint(len(self.names))]
        severity = np.random.randint(1, self.max_severity + 1)
        return degrade(image, name, severity, seed=np.random.randint(2**31))


def train_transform(config: DataConfig):
    """Augment training photos so the model sees more lighting and tilt.

    Signs are not mirrored: a flipped "keep right" is a different sign,
    so there is no horizontal flip here.
    """
    size = config.image_size
    augment = config.augment
    degrade = []
    if augment.degrade_probability > 0:
        degrade = [
            RandomDegrade(
                augment.degrade_names, augment.degrade_probability, augment.degrade_max_severity
            )
        ]
    return transforms.Compose(
        [
            *degrade,
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

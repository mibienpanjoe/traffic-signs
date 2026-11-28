"""Classical computer vision: HOG features and a linear SVM.

HOG (histogram of oriented gradients) describes a picture by its edges.
The image is turned to grayscale and its gradient is taken at every
pixel. The image is cut into cells of 8×8 pixels, and each cell keeps a
histogram of its gradient directions (9 bins over 0–180°), weighted by
gradient strength. Neighbouring 2×2 cells form a block, and each block's
histograms are normalized together (L2-Hys), so the description does not
change when the light gets brighter or dimmer. At 64×64 that is 7×7
blocks × 4 cells × 9 bins = 1,764 numbers per sign.

A linear SVM then draws one hyperplane per class (one-vs-rest) in that
1,764-dimensional space. Nothing here is learned from pixels: the
features are fixed, only the 43 hyperplanes are fit.
"""

import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.svm import LinearSVC

from signs.config import HogConfig


class HogTransform:
    """PIL image -> HOG feature vector (float32), for SignDataset."""

    def __init__(self, image_size: int, config: HogConfig):
        self.image_size = image_size
        self.config = config

    def image(self, image: Image.Image) -> np.ndarray:
        """The grayscale picture HOG reads, in [0, 1]."""
        gray = image.convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(gray, dtype=np.float32) / 255.0

    def __call__(self, image: Image.Image, visualize: bool = False):
        cells = self.config.pixels_per_cell
        blocks = self.config.cells_per_block
        return hog(
            self.image(image),
            orientations=self.config.orientations,
            pixels_per_cell=(cells, cells),
            cells_per_block=(blocks, blocks),
            block_norm="L2-Hys",
            feature_vector=True,
            visualize=visualize,
        )

    def feature_size(self) -> int:
        cells_across = self.image_size // self.config.pixels_per_cell
        blocks_across = cells_across - self.config.cells_per_block + 1
        per_block = self.config.cells_per_block**2 * self.config.orientations
        return blocks_across**2 * per_block


class HogSVM:
    """A fitted linear SVM on HOG features, with the transform it needs."""

    def __init__(self, transform: HogTransform, svm: LinearSVC):
        self.transform = transform
        self.svm = svm

    def decision(self, features: np.ndarray) -> np.ndarray:
        """One score per class; the largest is the prediction."""
        return self.svm.decision_function(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.svm.predict(features)

    def save(self, path: Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: Path) -> "HogSVM":
        # Only load files this project wrote: pickle runs code from the file.
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def fit_svm(features: np.ndarray, labels: np.ndarray, c: float, seed: int) -> LinearSVC:
    """One-vs-rest linear SVM. The primal solver suits more samples than features."""
    svm = LinearSVC(C=c, dual=False, max_iter=5000, random_state=seed)
    return svm.fit(features, labels)

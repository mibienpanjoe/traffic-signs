"""How close restoration brings a degraded image to the clean one, in PSNR and SSIM.

PSNR (peak signal-to-noise ratio) is 10·log10(1 / mean squared error),
in decibels: it counts every pixel's error the same. SSIM (structural
similarity) compares local means, contrasts and correlations, closer to
what a person notices. Both are measured against the clean image, at the
image's own size, before any resizing for a model.

A restoration that raises PSNR does not have to help a classifier: it
can smooth away the very edges the classifier reads. The benchmark with
--restore measures that second question; this module the first.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from signs.data.dataset import SignDataset
from signs.data.degradations import DEGRADATIONS, SEVERITIES, degrade, to_array
from signs.data.restoration import RESTORATIONS, restore

# Identical images have infinite PSNR; cap it so means stay finite.
PSNR_CAP = 60.0


def psnr(clean: np.ndarray, other: np.ndarray) -> float:
    if np.array_equal(clean, other):
        return PSNR_CAP
    return min(PSNR_CAP, float(peak_signal_noise_ratio(clean, other, data_range=1.0)))


def ssim(clean: np.ndarray, other: np.ndarray) -> float:
    side = min(clean.shape[:2])
    window = min(7, side if side % 2 else side - 1)
    return float(
        structural_similarity(clean, other, data_range=1.0, channel_axis=-1, win_size=window)
    )


def restoration_quality(data_dir, images: int = 500, seed: int = 0) -> list[dict]:
    """Mean PSNR and SSIM of degraded and of restored test images, per condition.

    The images are spread evenly over the test set. Each gets the same
    per-image seed as in the benchmark, (seed, row index).
    """
    test = SignDataset(data_dir, "test", transform=None)
    count = min(images, len(test))
    indices = np.linspace(0, len(test) - 1, count).round().astype(int)
    clean = {int(i): test[int(i)][0] for i in indices}
    rows = []
    for name, degradation in DEGRADATIONS.items():
        for severity in range(1, SEVERITIES + 1):
            scores = {
                "psnr_degraded": [],
                "psnr_restored": [],
                "ssim_degraded": [],
                "ssim_restored": [],
            }
            for index, image in clean.items():
                image_seed = (seed, index)
                degraded = degrade(image, name, severity, image_seed)
                restored = restore(degraded, name, severity, image_seed)
                reference = to_array(image)
                for kind, picture in (("degraded", degraded), ("restored", restored)):
                    scores[f"psnr_{kind}"].append(psnr(reference, to_array(picture)))
                    scores[f"ssim_{kind}"].append(ssim(reference, to_array(picture)))
            row = {
                "degradation": name,
                "severity": severity,
                "level": degradation.levels[severity - 1],
                "restoration": RESTORATIONS[name].title,
                "images": count,
            }
            row.update({key: round(float(np.mean(v)), 4) for key, v in scores.items()})
            rows.append(row)
            print(
                f"{name:>14} {severity}  PSNR {row['psnr_degraded']:5.1f} -> "
                f"{row['psnr_restored']:5.1f}  SSIM {row['ssim_degraded']:.3f} -> "
                f"{row['ssim_restored']:.3f}"
            )
    return rows

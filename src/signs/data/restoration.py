"""Classical restoration: undo a degradation before the classifier sees the image.

Each degradation of `degradations.py` gets the classical method made for
it. The restoration is told which degradation was applied and how strong
it was, and for blur it is even given the exact kernel. A real camera
would have to estimate all of that, so these numbers are the best case
for restoration: if it does not help here, it will not help in practice.

| Degradation    | Restoration |
|----------------|-------------|
| Motion blur    | Wiener deconvolution with the known kernel |
| Defocus blur   | Wiener deconvolution with the known kernel |
| Sensor noise   | Non-local means |
| Low light      | Non-local means, then CLAHE |
| Overexposure   | Inverse exposure (what clipped stays lost) |
| Low resolution | Unsharp masking |
| JPEG           | Bilateral filter, to smooth the 8×8 block edges |
| Dust haze      | Dark channel prior dehazing |
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import exposure, restoration

from signs.data.degradations import (
    DEGRADATIONS,
    check,
    known_blur_kernel,
    to_array,
    to_image,
    to_linear,
    to_srgb,
)

# Noise-to-signal ratio assumed by the Wiener filter. Larger values give
# up sharpness to avoid amplifying noise and ringing. 0.03 gave the best
# mean PSNR on 150 training images over both blurs and all severities
# (tried 0.01 to 0.1); test images were not used to pick it.
WIENER_BALANCE = 0.03


def wiener_deconvolve(linear: np.ndarray, kernel: np.ndarray, balance=WIENER_BALANCE):
    """Invert a known blur in the frequency domain: Ĵ = H* · I / (|H|² + K).

    Where the blur kept a frequency (|H| large), it is divided back out.
    Where the blur erased it (|H| near 0), the filter gives up instead of
    amplifying noise, which K controls. The image is padded by reflection
    first, so the filter does not treat it as a tile that wraps around.
    """
    pad = max(kernel.shape)
    padded = np.pad(linear, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    height, width = padded.shape[:2]
    # Kernel centred at (0, 0), as the FFT expects.
    psf = np.zeros((height, width))
    kh, kw = kernel.shape
    psf[:kh, :kw] = kernel
    psf = np.roll(psf, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    transfer = np.fft.fft2(psf)
    gain = np.conj(transfer) / (np.abs(transfer) ** 2 + balance)
    restored = np.stack(
        [np.real(np.fft.ifft2(np.fft.fft2(padded[..., c]) * gain)) for c in range(3)], axis=-1
    )
    return restored[pad:-pad, pad:-pad]


def noise_sigma(image: np.ndarray) -> float:
    """Standard deviation of the noise, estimated from the image (Immerkær, 1996).

    The 3×3 mask below is the difference of two Laplacians: it cancels
    flat regions and straight edges almost entirely, so what it leaves is
    mostly noise. Its mean absolute response, rescaled, estimates σ.
    """
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float)
    # Per channel, then averaged: averaging the channels first would also
    # average away part of the noise.
    estimates = []
    for c in range(image.shape[2]):
        response = ndimage.convolve(image[..., c], mask, mode="reflect")[1:-1, 1:-1]
        estimates.append(np.sqrt(np.pi / 2) * np.abs(response).mean() / 6)
    return float(np.mean(estimates))


def non_local_means(image: np.ndarray) -> np.ndarray:
    """Average each pixel with pixels whose surrounding patch looks alike.

    The noise level is estimated from the image itself, and patches of a
    sign repeat (the ring, the background), so noise averages out while
    edges, which only match other edges, are kept.
    """
    sigma = noise_sigma(image)
    return restoration.denoise_nl_means(
        image,
        h=0.8 * sigma,
        sigma=sigma,
        patch_size=5,
        patch_distance=6,
        fast_mode=True,
        channel_axis=-1,
    )


def clahe(image: np.ndarray) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization on brightness.

    The image is cut into tiles, and each tile's brightness histogram is
    spread over the full range, with a limit so noise is not blown up.
    """
    return exposure.equalize_adapthist(np.clip(image, 0, 1), clip_limit=0.02)


def dark_channel_dehaze(image: np.ndarray, omega=0.95, t_min=0.1) -> np.ndarray:
    """Dark channel prior (He, Sun and Tang, 2009), on linear light.

    In a clear outdoor photo, most patches have a pixel that is dark in at
    least one colour channel. Haze lifts those dark pixels towards the
    airlight A, so the patch minimum measures 1 − t. Then
    J = (I − A) / max(t, t_min) + A.
    A is taken from the haziest pixels: the brightest 0.1% of the dark channel.
    """
    linear = to_linear(image)
    patch = max(3, round(0.08 * min(linear.shape[:2])) | 1)
    dark = ndimage.minimum_filter(linear.min(axis=2), size=patch)
    count = max(1, dark.size // 1000)
    haziest = np.argsort(dark, axis=None)[-count:]
    airlight = linear.reshape(-1, 3)[haziest].mean(axis=0).clip(1e-3, 1)
    normalized = ndimage.minimum_filter((linear / airlight).min(axis=2), size=patch)
    # Smooth the transmission so patch edges do not leave halos.
    transmission = ndimage.gaussian_filter(1 - omega * normalized, sigma=patch / 2)
    clear = (linear - airlight) / np.maximum(transmission, t_min)[..., None] + airlight
    return to_srgb(clear)


def unsharp_mask(image: np.ndarray, sigma: float, amount: float = 1.0) -> np.ndarray:
    """Sharpen: I + amount · (I − blur(I)).

    The difference with a blurred copy holds the fine detail; adding it
    back makes edges steeper. It cannot recreate detail that was lost,
    only exaggerate what is left.
    """
    blurred = ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0))
    return np.clip(image + amount * (image - blurred), 0, 1)


def _deblur(image, name, severity, seed):
    kernel = known_blur_kernel(name, severity, image.shape, seed)
    return to_srgb(wiener_deconvolve(to_linear(image), kernel))


def _denoise(image, name, severity, seed):
    return non_local_means(image)


def _low_light(image, name, severity, seed):
    return clahe(non_local_means(image))


def _overexposure(image, name, severity, seed):
    exposure_level = DEGRADATIONS[name].levels[severity - 1]
    return to_srgb(to_linear(image) / exposure_level)


def _sharpen(image, name, severity, seed):
    factor = DEGRADATIONS[name].levels[severity - 1]
    return unsharp_mask(image, sigma=factor / 2)


def _deblock(image, name, severity, seed):
    # No setting tried on training images (colour σ 0.02 to 0.08) raised
    # PSNR on JPEG images; this mild one costs about 1 dB. Whether removing
    # block edges helps the classifier anyway is what the benchmark asks.
    return restoration.denoise_bilateral(
        image, sigma_color=0.03, sigma_spatial=0.8, channel_axis=-1
    )


def _dehaze(image, name, severity, seed):
    return dark_channel_dehaze(image)


@dataclass(frozen=True)
class Restoration:
    title: str
    apply: object


RESTORATIONS = {
    "motion_blur": Restoration("Wiener, known kernel", _deblur),
    "defocus_blur": Restoration("Wiener, known kernel", _deblur),
    "sensor_noise": Restoration("Non-local means", _denoise),
    "low_light": Restoration("Non-local means + CLAHE", _low_light),
    "overexposure": Restoration("Inverse exposure", _overexposure),
    "low_resolution": Restoration("Unsharp mask", _sharpen),
    "jpeg": Restoration("Bilateral filter", _deblock),
    "haze": Restoration("Dark channel prior", _dehaze),
}


def restore(image: Image.Image, name: str, severity: int, seed=0) -> Image.Image:
    """Undo `degrade(image, name, severity, seed)` as well as a classical method can.

    The same name, severity and seed as the degradation are needed: blur
    restoration rebuilds the exact kernel from them. Severity 0 returns
    the image unchanged.
    """
    check(name, severity)
    if severity == 0:
        return image.convert("RGB")
    restored = RESTORATIONS[name].apply(to_array(image), name, severity, seed)
    return to_image(restored)

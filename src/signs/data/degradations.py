"""Image degradations, each built from how a camera forms the picture.

A camera collects light (linear in the number of photons), blurs it
through the lens and the motion of the scene, counts photons with noise,
then encodes the result with a gamma curve and often JPEG. Each function
here undoes one step of a perfect camera, in the order where it happens:
optical effects (blur, haze, exposure) act on linear light, sensor noise
on photon counts, and resolution and JPEG on the final pixels.

Every degradation has 5 severities, from 1 (mild) to 5 (severe).
Severity 0 is the clean image. Random choices (blur direction, noise)
come from a seeded generator, so the same image, severity and seed give
the same pixels.

Blur sizes are a fraction of the crop's shorter side, not a pixel count.
The GTSRB crops range from 26 to 266 pixels across, so a blur of fixed
pixels would destroy a far sign and leave a near one untouched. As a
fraction, one severity means the same visible damage on every sign.

Gamma is the plain power law 2.2, close to sRGB and simpler to read.
"""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

SEVERITIES = 5
GAMMA = 2.2
# Read noise of the sensor, as a share of full white, the same for every shot.
READ_NOISE = 0.01
# Photons that make full white at normal exposure, for the low-light model.
FULL_WELL = 1000
# Dusty airlight: a dry, yellow-brown sky full of fine sand, in sRGB.
DUST_AIRLIGHT = (0.80, 0.68, 0.50)


def to_array(image: Image.Image) -> np.ndarray:
    """RGB PIL image -> float array H×W×3 in [0, 1]."""
    return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8))


def to_linear(srgb: np.ndarray) -> np.ndarray:
    """Pixel values -> light intensity. Pixels are gamma-encoded, light is not."""
    return np.clip(srgb, 0, 1) ** GAMMA


def to_srgb(linear: np.ndarray) -> np.ndarray:
    return np.clip(linear, 0, 1) ** (1 / GAMMA)


def line_kernel(length: float, angle_degrees: float) -> np.ndarray:
    """Point-spread function of straight motion: a line of `length` pixels.

    During the exposure, each scene point slides along a line, so its light
    is spread evenly over that line. The line is drawn with many sub-pixel
    samples, so a length of 3.4 pixels differs from 3.
    """
    if length <= 1:
        return np.ones((1, 1))
    half = int(np.ceil(length / 2)) + 1
    kernel = np.zeros((2 * half + 1, 2 * half + 1))
    angle = np.deg2rad(angle_degrees)
    for t in np.linspace(-length / 2, length / 2, int(length * 8) + 1):
        x = half + t * np.cos(angle)
        y = half - t * np.sin(angle)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        dx, dy = x - x0, y - y0
        # Share each sample between its four neighbouring pixels.
        kernel[y0, x0] += (1 - dx) * (1 - dy)
        kernel[y0, x0 + 1] += dx * (1 - dy)
        kernel[y0 + 1, x0] += (1 - dx) * dy
        kernel[y0 + 1, x0 + 1] += dx * dy
    return kernel / kernel.sum()


def disk_kernel(radius: float) -> np.ndarray:
    """Point-spread function of a lens out of focus: a uniform disk.

    Each scene point lands as a circle of confusion with this radius.
    Pixels on the rim are covered in part, measured with 8×8 sub-samples.
    """
    if radius < 0.5:
        return np.ones((1, 1))
    half = int(np.ceil(radius))
    size = 2 * half + 1
    # Sub-sample centres, 8 per pixel along each axis, relative to the kernel centre.
    positions = (np.arange(size * 8) + 0.5) / 8 - half - 0.5
    inside = positions[:, None] ** 2 + positions[None, :] ** 2 <= radius**2
    kernel = inside.reshape(size, 8, size, 8).mean(axis=(1, 3))
    return kernel / kernel.sum()


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Blur each channel of linear light with the same point-spread function."""
    return np.stack(
        [ndimage.convolve(image[..., c], kernel, mode="reflect") for c in range(3)], axis=-1
    )


def poisson_gaussian(
    linear: np.ndarray, photons: float, read_noise: float, rng: np.random.Generator
) -> np.ndarray:
    """Sensor noise: shot noise from counting photons, plus read noise.

    A pixel that should collect λ photons collects Poisson(λ): its noise
    grows as √λ, so dark pixels are relatively noisier. The electronics
    then add Gaussian read noise of the same size everywhere.
    `photons` is the count that makes full white.
    """
    counts = rng.poisson(np.clip(linear, 0, None) * photons)
    return counts / photons + rng.normal(0.0, read_noise, size=linear.shape)


def blur_kernel(name: str, fraction: float, shape, rng) -> np.ndarray:
    """The point-spread function a blur uses on an image of this shape.

    Motion blur draws its direction from `rng` first, so a generator in the
    same state gives the same kernel: `known_blur_kernel` relies on that.
    """
    side = min(shape[:2])
    if name == "motion_blur":
        return line_kernel(fraction * side, rng.uniform(0, 180))
    if name == "defocus_blur":
        return disk_kernel(fraction * side)
    raise ValueError(f"{name} is not a blur")


def motion_blur(image, length_fraction, rng):
    """I = J ∗ line(length, angle), with a random direction per image."""
    kernel = blur_kernel("motion_blur", length_fraction, image.shape, rng)
    return to_srgb(convolve(to_linear(image), kernel))


def defocus_blur(image, radius_fraction, rng):
    """I = J ∗ disk(radius)."""
    kernel = blur_kernel("defocus_blur", radius_fraction, image.shape, rng)
    return to_srgb(convolve(to_linear(image), kernel))


def sensor_noise(image, photons, rng):
    """I = Poisson(J·N)/N + 𝒩(0, σ²), with N photons at full white."""
    return to_srgb(poisson_gaussian(to_linear(image), photons, READ_NOISE, rng))


def low_light(image, exposure, rng):
    """Less light reaches the sensor, the camera amplifies what it got.

    I = g · [Poisson(e·J·N)/N + 𝒩(0, σ²)], with exposure e < 1 and gain
    g = e^(-1/2): the camera's gain makes up half of the lost light, in
    stops. The picture stays darker than normal, and its noise is
    amplified with it.
    """
    dark = poisson_gaussian(exposure * to_linear(image), FULL_WELL, READ_NOISE, rng)
    return to_srgb(dark * exposure**-0.5)


def overexposure(image, exposure, rng):
    """I = min(e·J, 1), with e > 1: bright parts saturate to white and their detail is lost."""
    return to_srgb(exposure * to_linear(image))


def low_resolution(image, factor, rng):
    """Average blocks of `factor`×`factor` pixels, then enlarge back.

    The same sign seen `factor` times further away: fewer pixels on it.
    Shrinking with an averaging filter avoids aliasing; enlarging with
    bilinear interpolation cannot bring the lost detail back.
    """
    height, width = image.shape[:2]
    small_size = (max(2, round(width / factor)), max(2, round(height / factor)))
    small = to_image(image).resize(small_size, Image.BOX)
    return to_array(small.resize((width, height), Image.BILINEAR))


def jpeg(image, quality, rng):
    """A real JPEG encoding and decoding, with PIL's default 4:2:0 color subsampling.

    JPEG cuts the image into 8×8 blocks and drops the fine frequencies of
    each one, more as the quality goes down: blocks and ringing appear.
    """
    buffer = io.BytesIO()
    to_image(image).save(buffer, format="JPEG", quality=int(quality))
    return to_array(Image.open(buffer))


def haze(image, transmission, rng):
    """Atmospheric scattering: I = J·t + A·(1 − t).

    A share t of the sign's light reaches the camera; the rest is replaced
    by airlight A, light scattered by dust towards the camera. Over a
    crop the sign is at one distance, so t is the same everywhere.
    """
    airlight = to_linear(np.array(DUST_AIRLIGHT))
    return to_srgb(to_linear(image) * transmission + airlight * (1 - transmission))


@dataclass(frozen=True)
class Degradation:
    name: str
    title: str
    parameter: str
    levels: tuple
    apply: object

    def __post_init__(self):
        if len(self.levels) != SEVERITIES:
            raise ValueError(f"{self.name} needs {SEVERITIES} levels, got {len(self.levels)}")


DEGRADATIONS = {
    d.name: d
    for d in (
        Degradation(
            "motion_blur",
            "Motion blur",
            "length / shorter side",
            (0.05, 0.10, 0.15, 0.20, 0.25),
            motion_blur,
        ),
        Degradation(
            "defocus_blur",
            "Defocus blur",
            "radius / shorter side",
            (0.02, 0.04, 0.06, 0.08, 0.10),
            defocus_blur,
        ),
        Degradation(
            "sensor_noise",
            "Sensor noise",
            "photons at full white",
            (400, 150, 60, 25, 10),
            sensor_noise,
        ),
        Degradation("low_light", "Low light", "exposure", (0.4, 0.2, 0.1, 0.05, 0.025), low_light),
        Degradation("overexposure", "Overexposure", "exposure", (1.5, 2, 3, 4.5, 7), overexposure),
        Degradation(
            "low_resolution", "Low resolution", "shrink factor", (1.5, 2, 3, 4, 6), low_resolution
        ),
        Degradation("jpeg", "JPEG", "quality", (40, 25, 15, 10, 5), jpeg),
        Degradation("haze", "Dust haze", "transmission", (0.9, 0.75, 0.6, 0.45, 0.3), haze),
    )
}


def check(name: str, severity: int) -> Degradation:
    if name not in DEGRADATIONS:
        raise ValueError(f"unknown degradation {name!r}; known: {', '.join(DEGRADATIONS)}")
    if not 0 <= severity <= SEVERITIES:
        raise ValueError(f"severity must be between 0 and {SEVERITIES}, got {severity}")
    return DEGRADATIONS[name]


def degrade(image: Image.Image, name: str, severity: int, seed=0) -> Image.Image:
    """The image under one degradation. Severity 0 returns it unchanged.

    `seed` may be a number or a tuple of numbers, such as (seed, index),
    so every image of a dataset gets its own repeatable randomness.
    """
    degradation = check(name, severity)
    if severity == 0:
        return image.convert("RGB")
    rng = np.random.default_rng(seed)
    level = degradation.levels[severity - 1]
    return to_image(degradation.apply(to_array(image), level, rng))


def known_blur_kernel(name: str, severity: int, shape, seed=0) -> np.ndarray:
    """The exact kernel `degrade(image, name, severity, seed)` blurred with.

    Restoration that is given this kernel knows more than a real camera
    would: it is the best case for deblurring.
    """
    degradation = check(name, severity)
    return blur_kernel(name, degradation.levels[severity - 1], shape, np.random.default_rng(seed))


class DegradedDataset:
    """A dataset of PIL images, degraded, then passed to the model's transform.

    `base` must return untransformed PIL images. Each image is degraded
    with its own seed, (seed, row index), so its noise and blur direction
    do not depend on batching or worker count. With `restore`, the
    matching classical restoration is applied after the degradation.
    """

    def __init__(
        self, base, name: str, severity: int, transform, seed: int = 0, restore: bool = False
    ):
        check(name, severity)
        if base.transform is not None:
            raise ValueError("the base dataset must return untransformed images")
        self.base = base
        self.labels = [base.labels[i] for i in base.indices]
        self.name = name
        self.severity = severity
        self.transform = transform
        self.seed = seed
        self.restore = restore

    def __len__(self):
        return len(self.base)

    def __getitem__(self, position):
        image, label = self.base[position]
        seed = (self.seed, self.base.indices[position])
        image = degrade(image, self.name, self.severity, seed)
        if self.restore:
            from signs.data.restoration import restore

            image = restore(image, self.name, self.severity, seed)
        return self.transform(image), label

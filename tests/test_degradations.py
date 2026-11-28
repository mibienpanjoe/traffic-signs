import numpy as np
import pytest
from PIL import Image

from signs.cli import main
from signs.data.dataset import SignDataset
from signs.data.degradations import (
    DEGRADATIONS,
    READ_NOISE,
    SEVERITIES,
    DegradedDataset,
    degrade,
    disk_kernel,
    haze,
    line_kernel,
    overexposure,
    poisson_gaussian,
    to_array,
)


@pytest.fixture(scope="module")
def sign():
    """A 64×64 picture with edges and colours: a red ring with a dark bar on grey."""
    y, x = np.mgrid[:64, :64]
    radius = np.hypot(x - 32, y - 32)
    pixels = np.full((64, 64, 3), 150, dtype=np.uint8)
    pixels[(radius > 20) & (radius < 28)] = (200, 30, 30)
    pixels[(abs(x - 32) < 10) & (abs(y - 32) < 4)] = (20, 20, 20)
    return Image.fromarray(pixels)


def _damage(image, clean):
    return float(np.abs(to_array(image) - to_array(clean)).mean())


@pytest.mark.parametrize("name", DEGRADATIONS)
def test_severity_zero_is_the_clean_image(sign, name):
    assert np.array_equal(np.asarray(degrade(sign, name, 0)), np.asarray(sign))


@pytest.mark.parametrize("name", DEGRADATIONS)
def test_damage_grows_with_severity(sign, name):
    damage = [_damage(degrade(sign, name, s, seed=3), sign) for s in range(SEVERITIES + 1)]
    assert damage[1] > 0
    # Each step is at least as bad as the one before, up to noise.
    steps = zip(damage, damage[1:], strict=False)
    assert all(later >= earlier - 0.002 for earlier, later in steps), damage


@pytest.mark.parametrize("name", DEGRADATIONS)
def test_keeps_size_and_repeats_with_the_same_seed(sign, name):
    first = degrade(sign, name, 3, seed=(1, 7))
    assert first.size == sign.size and first.mode == "RGB"
    assert np.array_equal(np.asarray(first), np.asarray(degrade(sign, name, 3, seed=(1, 7))))


def test_noise_differs_between_seeds(sign):
    one = np.asarray(degrade(sign, "sensor_noise", 3, seed=0))
    two = np.asarray(degrade(sign, "sensor_noise", 3, seed=1))
    assert not np.array_equal(one, two)


def test_unknown_name_and_severity_are_errors(sign):
    with pytest.raises(ValueError, match="unknown degradation"):
        degrade(sign, "rain", 1)
    with pytest.raises(ValueError, match="severity"):
        degrade(sign, "jpeg", SEVERITIES + 1)


def test_kernels_keep_the_light():
    for kernel in (line_kernel(7.3, 30), disk_kernel(4.5), line_kernel(0.5, 0), disk_kernel(0.2)):
        assert kernel.sum() == pytest.approx(1)
        assert (kernel >= 0).all()


def test_disk_covers_its_area():
    radius = 6
    kernel = disk_kernel(radius)
    covered = (kernel / kernel.max()).sum()
    assert covered == pytest.approx(np.pi * radius**2, rel=0.02)
    assert np.allclose(kernel, kernel.T) and np.allclose(kernel, kernel[::-1])


def test_line_kernel_lies_along_its_angle():
    horizontal = line_kernel(9, 0)
    centre = horizontal.shape[0] // 2
    assert horizontal[centre].sum() > 0.9
    vertical = line_kernel(9, 90)
    assert vertical[:, centre].sum() > 0.9


def test_haze_follows_the_scattering_model():
    # On linear light, I = J·t + A·(1 − t); a scene equal to the airlight stays the same.
    airlight = np.array([0.80, 0.68, 0.50])
    scene = np.broadcast_to(airlight, (4, 4, 3))
    assert np.allclose(haze(scene, 0.4, None), scene)
    assert np.allclose(haze(scene * 0, 0.0, None), scene)


def test_overexposure_saturates_to_white():
    image = np.full((4, 4, 3), 0.9)
    assert np.allclose(overexposure(image, 3, None), 1.0)


def test_sensor_noise_has_shot_and_read_parts():
    rng = np.random.default_rng(0)
    level, photons = 0.5, 100
    samples = poisson_gaussian(np.full(200_000, level), photons, READ_NOISE, rng)
    assert samples.mean() == pytest.approx(level, abs=0.002)
    expected_std = np.sqrt(level / photons + READ_NOISE**2)
    assert samples.std() == pytest.approx(expected_std, rel=0.02)


def test_degraded_dataset_seeds_each_row(fake_data_dir):
    base = SignDataset(fake_data_dir, "test", transform=None, indices=[5, 2, 9])
    dataset = DegradedDataset(base, "sensor_noise", 4, transform=np.asarray, seed=1)
    assert dataset.labels == [base.labels[i] for i in (5, 2, 9)]
    # Row 2 gets the same noise whether it is read alone or among others.
    alone = DegradedDataset(
        SignDataset(fake_data_dir, "test", transform=None, indices=[2]),
        "sensor_noise",
        4,
        transform=np.asarray,
        seed=1,
    )
    assert np.array_equal(dataset[1][0], alone[0][0])
    assert not np.array_equal(dataset[0][0], dataset[1][0])


def test_degraded_dataset_needs_raw_images(fake_data_dir):
    base = SignDataset(fake_data_dir, "test", transform=np.asarray)
    with pytest.raises(ValueError, match="untransformed"):
        DegradedDataset(base, "jpeg", 1, transform=np.asarray)


def test_degrade_command_draws_the_grid(fake_data_dir, tmp_path):
    out = tmp_path / "grid.png"
    main(
        [
            "degrade",
            "--set",
            f"data.data_dir={fake_data_dir}",
            "--min-size",
            "0",
            "--out",
            str(out),
        ]
    )
    assert out.is_file()
    with pytest.raises(SystemExit):
        main(["degrade", "--set", f"data.data_dir={fake_data_dir}", "--out", str(out)])

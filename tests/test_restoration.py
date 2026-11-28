import csv

import numpy as np
import pytest
from PIL import Image

from signs.analysis.restoration_quality import psnr, restoration_quality
from signs.analysis.robustness import RESTORED_FILE, benchmark_run, summarize
from signs.cli import main
from signs.data.dataset import SignDataset
from signs.data.degradations import (
    DEGRADATIONS,
    DegradedDataset,
    convolve,
    degrade,
    known_blur_kernel,
    to_array,
    to_linear,
    to_srgb,
)
from signs.data.restoration import (
    RESTORATIONS,
    dark_channel_dehaze,
    noise_sigma,
    non_local_means,
    restore,
    unsharp_mask,
)
from signs.training.trainer import train


@pytest.fixture(scope="module")
def sign():
    """A 64×64 sign-like picture: red ring, dark bar, grey background."""
    y, x = np.mgrid[:64, :64]
    radius = np.hypot(x - 32, y - 32)
    pixels = np.full((64, 64, 3), 150, dtype=np.uint8)
    pixels[(radius > 20) & (radius < 28)] = (200, 30, 30)
    pixels[(abs(x - 32) < 10) & (abs(y - 32) < 4)] = (20, 20, 20)
    return Image.fromarray(pixels)


def _gain(sign, name, severity, seed=0):
    clean = to_array(sign)
    damaged = degrade(sign, name, severity, seed)
    restored = restore(damaged, name, severity, seed)
    return psnr(clean, to_array(restored)) - psnr(clean, to_array(damaged))


def test_every_degradation_has_a_restoration():
    assert set(RESTORATIONS) == set(DEGRADATIONS)


@pytest.mark.parametrize("name", DEGRADATIONS)
def test_restore_keeps_size_repeats_and_skips_clean(sign, name):
    damaged = degrade(sign, name, 3, seed=(0, 4))
    first = restore(damaged, name, 3, seed=(0, 4))
    assert first.size == sign.size and first.mode == "RGB"
    assert np.array_equal(np.asarray(first), np.asarray(restore(damaged, name, 3, seed=(0, 4))))
    assert np.array_equal(np.asarray(restore(sign, name, 0)), np.asarray(sign))


@pytest.mark.parametrize("name", ["motion_blur", "defocus_blur"])
def test_known_kernel_is_the_one_used(sign, name):
    kernel = known_blur_kernel(name, 4, (64, 64, 3), seed=(2, 9))
    expected = to_srgb(convolve(to_linear(to_array(sign)), kernel))
    damaged = to_array(degrade(sign, name, 4, seed=(2, 9)))
    assert np.abs(damaged - expected).max() <= 1 / 255 + 1e-9


@pytest.mark.parametrize("name", ["motion_blur", "defocus_blur", "sensor_noise", "overexposure"])
def test_restoration_brings_the_image_closer(sign, name):
    assert _gain(sign, name, 3) > 1.0


def test_inverse_exposure_is_exact_where_nothing_clipped():
    # A dark scene: at exposure 2 nothing reaches white, so dividing back is exact.
    dark = Image.fromarray(np.full((16, 16, 3), 60, dtype=np.uint8))
    restored = restore(degrade(dark, "overexposure", 2), "overexposure", 2)
    assert np.abs(np.asarray(restored, dtype=int) - 60).max() <= 2


def test_noise_estimate_matches_the_noise():
    rng = np.random.default_rng(0)
    flat = np.full((80, 80, 3), 0.5) + rng.normal(0, 0.05, (80, 80, 3))
    assert noise_sigma(flat) == pytest.approx(0.05, rel=0.1)
    assert noise_sigma(np.full((80, 80, 3), 0.5)) == pytest.approx(0)


def test_non_local_means_removes_noise(sign):
    clean = to_array(sign)
    noisy = clean + np.random.default_rng(1).normal(0, 0.06, clean.shape)
    assert psnr(clean, np.clip(non_local_means(noisy), 0, 1)) > psnr(clean, np.clip(noisy, 0, 1))


def test_dark_channel_dehazes_a_scene_with_dark_pixels(sign):
    # The sign has dark pixels (the bar), which is what the prior needs.
    clean = to_array(sign)
    hazy = to_array(degrade(sign, "haze", 4))
    assert psnr(clean, dark_channel_dehaze(hazy)) > psnr(clean, hazy)


def test_unsharp_mask_steepens_edges_and_stays_in_range():
    step = np.zeros((20, 20, 3))
    step[:, 10:] = 0.5
    sharp = unsharp_mask(step, sigma=1.5)
    assert sharp.min() >= 0 and sharp.max() <= 1
    assert sharp[:, 10].mean() > 0.5  # overshoot on the bright side of the edge


def test_degraded_dataset_restores_when_asked(fake_data_dir):
    base = SignDataset(fake_data_dir, "test", transform=None, indices=[0, 1])
    plain = DegradedDataset(base, "sensor_noise", 4, np.asarray)
    restored = DegradedDataset(base, "sensor_noise", 4, np.asarray, restore=True)
    assert not np.array_equal(plain[0][0], restored[0][0])


def test_benchmark_with_restoration_writes_its_own_file(tiny_config):
    tiny_config.train.epochs = 1
    train(tiny_config)
    main(
        [
            "benchmark",
            str(tiny_config.run_dir),
            "--restore",
            "--degradations",
            "jpeg",
            "--severities",
            "2",
        ]
    )
    path = tiny_config.run_dir / RESTORED_FILE
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["degradation"] for row in rows] == ["clean", "jpeg"]
    assert not (tiny_config.run_dir / "robustness.csv").exists()
    # A CSV path is read as given, so restored and plain tables can be compared.
    summary = summarize({"restored": [path]})
    assert len(summary) == 2
    benchmark_run(tiny_config.run_dir, ["jpeg"], [2])
    plain = summarize({"plain": [tiny_config.run_dir]})
    assert plain[0]["accuracy_mean"] == summary[0]["accuracy_mean"]  # same clean images


def test_restoration_quality_measures_both_images(fake_data_dir, tmp_path):
    rows = restoration_quality(fake_data_dir, images=4)
    assert len(rows) == len(DEGRADATIONS) * 5
    for row in rows:
        assert row["images"] == 4
        assert 0 <= row["ssim_restored"] <= 1
        assert row["psnr_degraded"] <= 60
    out = tmp_path / "quality.csv"
    main(
        [
            "restoration-quality",
            "--set",
            f"data.data_dir={fake_data_dir}",
            "--images",
            "2",
            "--out",
            str(out),
        ]
    )
    assert out.is_file()


def test_restoration_figure_is_drawn(fake_data_dir, tmp_path):
    out = tmp_path / "restorations.png"
    main(
        [
            "degrade",
            "--restore",
            "--set",
            f"data.data_dir={fake_data_dir}",
            "--min-size",
            "0",
            "--out",
            str(out),
        ]
    )
    assert out.is_file()

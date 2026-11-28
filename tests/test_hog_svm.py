import json

import numpy as np
import pytest
from PIL import Image

from signs.config import HogConfig, load_config
from signs.evaluate import evaluate
from signs.models.hog_svm import HogTransform
from signs.predict import predict
from signs.training.trainer import train


def test_feature_size_matches_the_cell_layout():
    transform = HogTransform(64, HogConfig())
    image = Image.fromarray(np.zeros((40, 50, 3), dtype=np.uint8))
    features = transform(image)
    # 8x8 cells -> 7x7 blocks of 2x2 cells x 9 bins
    assert features.shape == (7 * 7 * 2 * 2 * 9,) == (transform.feature_size(),)


def test_hog_follows_an_edge_and_ignores_brightness():
    transform = HogTransform(32, HogConfig(pixels_per_cell=8))
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[:, 16:] = 120
    dim = transform(Image.fromarray(pixels))
    bright = transform(Image.fromarray((pixels * 2).astype(np.uint8)))
    assert dim.max() > 0
    # Block normalization: doubling the contrast leaves the features alone.
    assert np.allclose(dim, bright, atol=1e-3)


def test_hog_svm_trains_evaluates_and_predicts(fake_data_dir, tmp_path):
    config = load_config(
        None,
        [
            f"data.data_dir={fake_data_dir}",
            "data.split=random",
            "data.image_size=24",
            "model.name=hog_svm",
            "model.pretrained=false",
            "model.freeze_features=false",
            "model.hog.svm_c=[0.1, 1.0]",
            f"runs_dir={tmp_path}",
            "run_name=hog",
        ],
    )
    summary = train(config)
    assert summary["best_svm_c"] in (0.1, 1.0)
    run_dir = config.run_dir
    assert (run_dir / "model.pkl").is_file()
    assert len((run_dir / "history.csv").read_text().splitlines()) == 3

    evaluate(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["test"]["images"] == 43

    image_path = tmp_path / "sign.png"
    Image.fromarray(np.full((24, 24, 3), 90, dtype=np.uint8)).save(image_path)
    top = predict(run_dir, image_path, top_k=3)
    scores = [score for _, score in top]
    assert len(top) == 3 and scores == sorted(scores, reverse=True)


def test_hog_figure_is_drawn(fake_track_dir, tmp_path):
    from signs.config import DataConfig
    from signs.data.dataset import preview_dataset
    from signs.viz.figures import plot_hog_examples

    dataset = preview_dataset(DataConfig(data_dir=str(fake_track_dir)))
    out = tmp_path / "hog.png"
    plot_hog_examples(dataset, HogTransform(32, HogConfig()), [0, 14], out)
    assert out.is_file()


def test_svm_c_accepts_one_number_and_rejects_nonsense():
    assert load_config(None, ["model.hog.svm_c=0.1"]).model.hog.svm_c == [0.1]
    for bad in ("[]", "[0, 1]", "fast"):
        with pytest.raises(ValueError, match="svm_c must be one positive number"):
            load_config(None, [f"model.hog.svm_c={bad}"])

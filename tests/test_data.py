import torch

from signs.config import DataConfig
from signs.data.dataset import load_splits, make_loader
from signs.data.splits import random_split_indices
from signs.data.transforms import eval_transform


def test_split_is_disjoint_complete_and_repeatable():
    train_a, val_a = random_split_indices(1000, 0.2, seed=42)
    train_b, val_b = random_split_indices(1000, 0.2, seed=42)
    assert (train_a, val_a) == (train_b, val_b)
    assert len(val_a) == 200
    assert not set(train_a) & set(val_a)
    assert sorted(train_a + val_a) == list(range(1000))


def test_split_changes_with_seed():
    assert random_split_indices(1000, 0.2, seed=1)[1] != random_split_indices(1000, 0.2, seed=2)[1]


def test_splits_load_with_expected_shapes(fake_data_dir):
    config = DataConfig(data_dir=str(fake_data_dir), split="random", image_size=32, batch_size=8)
    train_set, val_set, test_set = load_splits(config)
    assert len(train_set) + len(val_set) == 43 * 3
    assert len(test_set) == 43
    images, labels = next(iter(make_loader(train_set, config, shuffle=True, seed=0)))
    assert images.shape == (8, 3, 32, 32)
    assert labels.dtype == torch.int64


def test_eval_transform_is_deterministic(fake_data_dir):
    config = DataConfig(data_dir=str(fake_data_dir), split="random", image_size=32)
    _, _, test_set = load_splits(config)
    first, _ = test_set[0]
    second, _ = test_set[0]
    assert torch.equal(first, second)
    assert eval_transform(config) is not None


def test_seeded_loader_repeats_its_order(fake_data_dir):
    config = DataConfig(data_dir=str(fake_data_dir), split="random", image_size=32, batch_size=16)
    train_set, _, _ = load_splits(config)

    def first_labels(seed):
        _, labels = next(iter(make_loader(train_set, config, shuffle=True, seed=seed)))
        return labels.tolist()

    assert first_labels(3) == first_labels(3)
    assert first_labels(3) != first_labels(4)


def test_track_structure_finds_the_tracks(fake_track_dir):
    from signs.data.tracks import track_structure

    stats = track_structure(DataConfig(data_dir=str(fake_track_dir)))
    assert stats["tracks"] == 86
    assert stats["rows_divide_into_tracks"] and stats["one_class_per_track"]

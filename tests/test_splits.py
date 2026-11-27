import pytest

from signs.config import DataConfig, load_config
from signs.data.dataset import load_splits
from signs.data.splits import TRACK_LENGTH, track_ids, track_split_indices


def _labels(tracks_per_class):
    labels = []
    for label, count in enumerate(tracks_per_class):
        labels += [label] * (count * TRACK_LENGTH)
    return labels


def test_no_track_is_on_both_sides():
    labels = _labels([5, 50, 12, 7])
    train_idx, val_idx = track_split_indices(labels, 0.2, seed=0)
    tracks = track_ids(len(labels))
    assert not {tracks[i] for i in train_idx} & {tracks[i] for i in val_idx}
    assert sorted(train_idx + val_idx) == list(range(len(labels)))


def test_every_class_gets_its_share_of_tracks():
    labels = _labels([5, 50, 12, 7])
    _, val_idx = track_split_indices(labels, 0.2, seed=0)
    val_tracks_per_class = [0, 0, 0, 0]
    for i in val_idx[::TRACK_LENGTH]:
        val_tracks_per_class[labels[i]] += 1
    assert val_tracks_per_class == [1, 10, 2, 1]


def test_smallest_class_keeps_tracks_on_both_sides():
    labels = _labels([2, 2])
    train_idx, val_idx = track_split_indices(labels, 0.05, seed=0)
    assert {labels[i] for i in val_idx} == {0, 1}
    assert {labels[i] for i in train_idx} == {0, 1}


def test_track_split_is_repeatable_and_seeded():
    labels = _labels([10, 10, 10])
    assert track_split_indices(labels, 0.2, 1) == track_split_indices(labels, 0.2, 1)
    assert track_split_indices(labels, 0.2, 1) != track_split_indices(labels, 0.2, 2)


def test_rows_out_of_track_order_are_rejected():
    labels = _labels([2, 2])
    labels[10] = 1
    with pytest.raises(ValueError, match="mixes classes"):
        track_split_indices(labels, 0.2, seed=0)


def test_rows_that_do_not_fill_tracks_are_rejected():
    with pytest.raises(ValueError, match="do not divide into tracks"):
        track_split_indices([0] * 45, 0.2, seed=0)


def test_load_splits_uses_the_track_split_by_default(fake_track_dir):
    config = DataConfig(data_dir=str(fake_track_dir), image_size=32)
    assert config.split == "track"
    train_set, val_set, _ = load_splits(config)
    train_tracks = {i // TRACK_LENGTH for i in train_set.indices}
    val_tracks = {i // TRACK_LENGTH for i in val_set.indices}
    assert not train_tracks & val_tracks
    assert len(val_tracks) == 43  # 2 tracks per class: one each side


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="data.split must be track or random"):
        load_config(None, ["data.split=frames"])

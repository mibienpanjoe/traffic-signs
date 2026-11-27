"""Hold part of the training table out for validation.

GTSRB was cut from video. Each physical sign was filmed as a track of 30
consecutive frames, and the training table keeps them in order: rows
0-29 are one track, rows 30-59 the next, and so on (see `signs tracks`).
Frames of one track are almost the same picture. A random split puts
some of them in training and the rest in validation, so validation
measures recall of a sign the model has already seen. The track split
keeps every track whole on one side.
"""

from collections import defaultdict

import torch

TRACK_LENGTH = 30
SPLITS = ("track", "random")


def track_ids(n: int, track_length: int = TRACK_LENGTH) -> list[int]:
    """Track number of each row of the training table."""
    if n % track_length:
        raise ValueError(f"{n} rows do not divide into tracks of {track_length}")
    return [index // track_length for index in range(n)]


def random_split_indices(n: int, val_fraction: float, seed: int):
    """Shuffle indices 0..n-1 with a fixed seed and cut off a validation share.

    Returns (train_indices, val_indices). The same seed always gives the
    same split. Frames of one track end up on both sides.
    """
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n, generator=generator).tolist()
    n_val = int(n * val_fraction)
    return permutation[n_val:], permutation[:n_val]


def track_split_indices(
    labels: list[int], val_fraction: float, seed: int, track_length: int = TRACK_LENGTH
):
    """Send whole tracks to validation, the same share of tracks for every class.

    Each class keeps at least one track on each side, so a class with five
    tracks still appears in validation. Returns (train_indices, val_indices),
    both sorted.
    """
    tracks = track_ids(len(labels), track_length)
    tracks_of_class: dict[int, list[int]] = defaultdict(list)
    for track in range(len(labels) // track_length):
        start = track * track_length
        label = labels[start]
        if any(labels[i] != label for i in range(start, start + track_length)):
            raise ValueError(f"track {track} mixes classes; rows are not in track order")
        tracks_of_class[label].append(track)

    generator = torch.Generator().manual_seed(seed)
    val_tracks = set()
    for label in sorted(tracks_of_class):
        class_tracks = tracks_of_class[label]
        order = torch.randperm(len(class_tracks), generator=generator).tolist()
        n_val = round(len(class_tracks) * val_fraction)
        n_val = min(max(n_val, 1), len(class_tracks) - 1)
        val_tracks.update(class_tracks[i] for i in order[:n_val])

    train_idx = [i for i, track in enumerate(tracks) if track not in val_tracks]
    val_idx = [i for i, track in enumerate(tracks) if track in val_tracks]
    return train_idx, val_idx


def split_indices(labels: list[int], method: str, val_fraction: float, seed: int):
    if method == "track":
        return track_split_indices(labels, val_fraction, seed)
    if method == "random":
        return random_split_indices(len(labels), val_fraction, seed)
    raise ValueError(f"unknown split {method!r}; known: {list(SPLITS)}")

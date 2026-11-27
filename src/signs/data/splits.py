"""Hold part of the training table out for validation."""

import torch


def random_split_indices(n: int, val_fraction: float, seed: int):
    """Shuffle indices 0..n-1 with a fixed seed and cut off a validation share.

    Returns (train_indices, val_indices). The same seed always gives the
    same split.
    """
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n, generator=generator).tolist()
    n_val = int(n * val_fraction)
    return permutation[n_val:], permutation[:n_val]

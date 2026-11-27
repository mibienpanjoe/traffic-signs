"""Make a training run repeatable: same seed, same weights, same numbers."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch, and ask PyTorch for deterministic kernels.

    `warn_only=True` keeps a run going when an operation has no deterministic
    version on the current device, instead of stopping it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own, reproducible NumPy and Python seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loader_generator(seed: int) -> torch.Generator:
    """Generator for DataLoader shuffling, independent of the global RNG."""
    return torch.Generator().manual_seed(seed)

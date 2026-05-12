"""DDPM training loop.

Source: HW08_20231049.ipynb code cell #31.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader


def training_loop(
    ddpm,
    loader: DataLoader,
    n_epochs: int,
    optim: torch.optim.Optimizer,
    device: torch.device,
    display: bool = False,
    store_path: str | Path = "ddpm_model.pt",
) -> list[float]:
    """Train a `MyDDPM` for `n_epochs` and persist the best checkpoint.

    Returns a per-epoch loss history.
    """
    raise NotImplementedError("Port from HW08 cell #31")

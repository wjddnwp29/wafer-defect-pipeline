"""Plotting helpers for samples and the forward diffusion process.

Source: HW08_20231049.ipynb code cells #11 (show_images), #17 (show_forward).
"""

from __future__ import annotations

import torch


def show_images(images: torch.Tensor, title: str = "", labels=None, class_names=None) -> None:
    """Plot a batch of images in a grid.

    Source: HW08 cell #11.
    """
    raise NotImplementedError("Port from HW08 cell #11")


def show_forward(ddpm, loader, device: torch.device) -> None:
    """Visualize the forward (noising) diffusion process for a sample batch.

    Source: HW08 cell #17.
    """
    raise NotImplementedError("Port from HW08 cell #17")

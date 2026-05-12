"""DDPM wrapper around the UNet.

Source: HW08_20231049.ipynb code cell #19.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MyDDPM(nn.Module):
    """DDPM with linear beta schedule and class-conditional UNet backbone.

    Source: HW08 cell #19.
    """

    def __init__(
        self,
        network: nn.Module,
        n_steps: int = 200,
        min_beta: float = 1e-4,
        max_beta: float = 0.02,
        device: torch.device | None = None,
        image_chw: tuple[int, int, int] = (1, 28, 28),
    ):
        super().__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device) if device else network
        # TODO (Phase 1 port): build betas / alphas / alpha_bars from HW08 cell #19.

    def forward(self, x0: torch.Tensor, t: torch.Tensor, eta: torch.Tensor | None = None):
        raise NotImplementedError("Port from HW08 cell #19")

    def backward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Port from HW08 cell #19")

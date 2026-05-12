from __future__ import annotations

import torch
import torch.nn as nn


class MyDDPM(nn.Module):

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
        self.network = network.to(device) if device is not None else network

        betas = torch.linspace(min_beta, max_beta, n_steps)
        if device is not None:
            betas = betas.to(device)
        self.betas = betas
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        eta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w, device=x0.device)

        noisy = (
            a_bar.sqrt().reshape(n, 1, 1, 1) * x0
            + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        )
        return noisy

    def backward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.network(x, t, y)

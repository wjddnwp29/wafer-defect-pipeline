"""Consistency Model and EMA helper.

Ported from HW08_20231049.ipynb (update_ema, ConsistencyModel).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def update_ema(
    target_model: nn.Module,
    student_model: nn.Module,
    decay: float = 0.99999,
) -> None:
    """In-place EMA update: theta_target = decay * theta_target + (1 - decay) * theta_student."""
    with torch.no_grad():
        for t_param, s_param in zip(
            target_model.parameters(), student_model.parameters(), strict=True
        ):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1 - decay)


class ConsistencyModel(nn.Module):
    """Consistency Model wrapper sharing the UNet backbone with the teacher DDPM."""

    def __init__(
        self,
        network: nn.Module,
        n_steps: int = 1000,
        min_beta: float = 1e-4,
        max_beta: float = 0.02,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.device = device
        self.network = network.to(device) if device is not None else network

        betas = torch.linspace(min_beta, max_beta, n_steps)
        if device is not None:
            betas = betas.to(device)
        self.betas = betas
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1 - self.alpha_bars[t]) / torch.sqrt(self.alpha_bars[t])

    def consistency_function(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        eps_pred = self.network(x, t, y)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        x0_pred = (x - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
        x0_pred = torch.tanh(x0_pred)
        return x0_pred

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.consistency_function(x, t, y)

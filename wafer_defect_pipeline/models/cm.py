"""Consistency Model and EMA helper.

Source: HW08_20231049.ipynb code cells #43-44.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def update_ema(target_model: nn.Module, student_model: nn.Module, decay: float = 0.99999) -> None:
    """In-place EMA update of target parameters from the student.

    Source: HW08 cell #43.
    """
    raise NotImplementedError("Port from HW08 cell #43")


class ConsistencyModel(nn.Module):
    """Consistency Model wrapper sharing the UNet backbone with the teacher DDPM.

    Source: HW08 cell #44.
    """

    def __init__(self, network: nn.Module, n_steps: int = 1000, device: torch.device | None = None):
        super().__init__()
        self.n_steps = n_steps
        self.device = device
        self.network = network.to(device) if device else network
        # TODO (Phase 1 port): noise + sigma schedule from HW08 cell #44.

    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Port from HW08 cell #44")

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Port from HW08 cell #44")

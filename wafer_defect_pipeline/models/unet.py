"""Class-conditional UNet used as the noise predictor for DDPM and the backbone for CM.

Source: HW08_20231049.ipynb code cells #20-22 (MyBlock, sinusoidal_embedding, MyUNet).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sinusoidal_embedding(n: int, d: int) -> torch.Tensor:
    """Standard sinusoidal positional embedding used for the diffusion timestep.

    TODO (Phase 1 port): copy implementation from HW08 cell #21.
    """
    raise NotImplementedError("Port from HW08 cell #21")


class MyBlock(nn.Module):
    """LN + two 3x3 convs + activation, used as a building block of the UNet.

    Source: HW08 cell #20.
    """

    def __init__(
        self,
        shape,
        in_c: int,
        out_c: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: nn.Module | None = None,
        normalize: bool = True,
    ):
        super().__init__()
        # TODO (Phase 1 port): copy implementation from HW08 cell #20.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Port from HW08 cell #20")


class MyUNet(nn.Module):
    """Class-conditional UNet for 28x28 wafer maps.

    Source: HW08 cell #22.
    """

    def __init__(self, n_steps: int = 1000, time_emb_dim: int = 100, num_classes: int = 8):
        super().__init__()
        self.n_steps = n_steps
        self.time_emb_dim = time_emb_dim
        self.num_classes = num_classes
        # TODO (Phase 1 port): copy embedding + encoder + bottleneck + decoder
        # from HW08 cell #22.

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Port from HW08 cell #22")

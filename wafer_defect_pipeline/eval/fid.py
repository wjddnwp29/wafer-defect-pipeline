"""Frechet Inception Distance using a pretrained Inception v3.

Source: HW08_20231049.ipynb code cells #28-29.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class InceptionFeatureExtractor(nn.Module):
    """Pretrained Inception v3 with the classification head replaced by Identity.

    Source: HW08 cell #28.
    """

    def __init__(self):
        super().__init__()
        # TODO (Phase 1 port): instantiate inception_v3 from HW08 cell #28.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Port from HW08 cell #28")


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Standard FID between two feature matrices (N, D).

    Source: HW08 cell #29.
    """
    raise NotImplementedError("Port from HW08 cell #29")

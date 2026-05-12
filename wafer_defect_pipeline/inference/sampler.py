"""Class-conditional samplers for DDPM, DDIM, and Consistency Models.

Source: HW08_20231049.ipynb code cells #32 (DDPM), #33 (DDIM), #47 (CM).
"""

from __future__ import annotations

import torch


def generate_ddpm_with_nfe(
    ddpm,
    n_samples: int = 16,
    device: torch.device | None = None,
    classes: torch.Tensor | None = None,
    c: int = 1,
    h: int = 28,
    w: int = 28,
):
    """Full-trajectory DDPM ancestral sampling.

    Source: HW08 cell #32.
    """
    raise NotImplementedError("Port from HW08 cell #32")


def generate_ddim_with_nfe(
    ddpm,
    n_samples: int = 16,
    device: torch.device | None = None,
    classes: torch.Tensor | None = None,
    n_steps: int = 50,
    eta: float = 0.0,
    c: int = 1,
    h: int = 28,
    w: int = 28,
):
    """Deterministic DDIM sampling with configurable step count.

    Source: HW08 cell #33.
    """
    raise NotImplementedError("Port from HW08 cell #33")


def generate_cm_with_nfe(
    cm,
    n_samples: int = 16,
    device: torch.device | None = None,
    classes: torch.Tensor | None = None,
    n_steps: int = 10,
    c: int = 1,
    h: int = 28,
    w: int = 28,
):
    """Multi-step Consistency Model sampling.

    Source: HW08 cell #47.
    """
    raise NotImplementedError("Port from HW08 cell #47")

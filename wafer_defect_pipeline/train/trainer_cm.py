"""Consistency Model distillation training.

Source: HW08_20231049.ipynb code cells #45-46.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def consistency_distillation_loss(teacher_ddpm, student_cm, target_cm, x0, y, device):
    """Multi-scale consistency distillation loss.

    Source: HW08 cell #45.
    """
    raise NotImplementedError("Port from HW08 cell #45")


def train_consistency_model(
    teacher_ddpm,
    dataloader: DataLoader,
    n_epochs: int,
    device: torch.device,
):
    """Distill a Consistency Model from a frozen teacher DDPM.

    Source: HW08 cell #46.
    """
    raise NotImplementedError("Port from HW08 cell #46")

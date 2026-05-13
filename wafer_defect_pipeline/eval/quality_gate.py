from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from wafer_defect_pipeline.eval.fid import calculate_fid, extract_features


@dataclass(frozen=True)
class GateResult:
    fid: float
    threshold: float
    passed: bool


def evaluate_fid_gate(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    threshold: float,
) -> GateResult:
    fid = calculate_fid(real_features, fake_features)
    return GateResult(fid=fid, threshold=threshold, passed=fid <= threshold)


def evaluate_fid_gate_from_images(
    real_images: torch.Tensor | np.ndarray,
    fake_images: torch.Tensor | np.ndarray,
    feature_extractor: nn.Module,
    device: torch.device,
    threshold: float,
    batch_size: int = 50,
) -> GateResult:
    real_features = extract_features(real_images, feature_extractor, device, batch_size)
    fake_features = extract_features(fake_images, feature_extractor, device, batch_size)
    return evaluate_fid_gate(real_features, fake_features, threshold)

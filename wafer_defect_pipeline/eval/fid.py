from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torchvision.models import Inception_V3_Weights, inception_v3

INCEPTION_INPUT_SIZE = 299


class InceptionFeatureExtractor(nn.Module):

    def __init__(self, weights: Inception_V3_Weights | None = Inception_V3_Weights.DEFAULT):
        super().__init__()
        self.inception = inception_v3(weights=weights, transform_input=False)
        self.inception.fc = nn.Identity()
        self.inception.eval()
        for param in self.inception.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(
            x, size=(INCEPTION_INPUT_SIZE, INCEPTION_INPUT_SIZE),
            mode="bilinear", align_corners=False,
        )
        x = (x + 1) / 2
        return self.inception(x)


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean))


def extract_features(
    images: torch.Tensor | np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 50,
) -> np.ndarray:
    if isinstance(images, np.ndarray):
        images_t = torch.from_numpy(images).float()
    else:
        images_t = images.float()
    if images_t.ndim == 3:
        images_t = images_t.unsqueeze(1)

    features = []
    for i in range(0, len(images_t), batch_size):
        batch = images_t[i : i + batch_size].to(device)
        with torch.no_grad():
            feat = model(batch).cpu().numpy()
        features.append(feat)
    return np.concatenate(features, axis=0)

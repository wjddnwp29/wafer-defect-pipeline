"""WM-811K dataset acquisition and standard preprocessing transform.

Ported from HW08_20231049.ipynb (kagglehub download + transform pipeline).
"""

from __future__ import annotations

from pathlib import Path

from torchvision import transforms

DEFAULT_IMG_SIZE = 28
DATASET_SLUG = "qingyi/wm811k-wafer-map"
PICKLE_FILENAME = "LSWMD.pkl"


def download_wm811k(cache_dir: str | Path | None = None) -> Path:
    """Download WM-811K via kagglehub and return the path to LSWMD.pkl.

    cache_dir, if given, overrides kagglehub's default location via KAGGLEHUB_CACHE.
    """
    import os

    import kagglehub

    if cache_dir is not None:
        os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)

    dataset_dir = Path(kagglehub.dataset_download(DATASET_SLUG))
    pkl = dataset_dir / PICKLE_FILENAME
    if not pkl.exists():
        matches = list(dataset_dir.rglob(PICKLE_FILENAME))
        if not matches:
            raise FileNotFoundError(
                f"{PICKLE_FILENAME} not found under {dataset_dir}"
            )
        pkl = matches[0]
    return pkl


def build_transform(img_size: int = DEFAULT_IMG_SIZE) -> transforms.Compose:
    """Standard preprocessing pipeline: resize -> tensor -> normalize to [-1, 1]."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

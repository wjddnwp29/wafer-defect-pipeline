from __future__ import annotations

from pathlib import Path

from torchvision import transforms

DEFAULT_IMG_SIZE = 28
DATASET_SLUG = "qingyi/wm811k-wafer-map"
PICKLE_FILENAME = "LSWMD.pkl"


def download_wm811k(cache_dir: str | Path | None = None) -> Path:
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
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

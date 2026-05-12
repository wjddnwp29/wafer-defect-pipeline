"""WM-811K dataset acquisition via kagglehub.

Source: HW08_20231049.ipynb code cell #1 (dataset download).
"""

from __future__ import annotations

from pathlib import Path


def download_wm811k(cache_dir: str | Path | None = None) -> Path:
    """Download WM-811K via kagglehub and return the local path.

    TODO (Phase 1 port):
      - Move the kagglehub.dataset_download() call from HW08 cell #1 here.
      - Resolve the pickle file path inside the downloaded directory.
      - Allow overriding the cache directory via env var or argument.
    """
    raise NotImplementedError("Port from HW08 cell #1")

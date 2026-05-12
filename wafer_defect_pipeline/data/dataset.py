"""Class-conditional WM-811K wafer map dataset.

Source: HW08_20231049.ipynb code cell #4.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset


class ConditionalWaferDataset(Dataset):
    """Wafer map dataset that yields (image, class_index) pairs.

    TODO (Phase 1 port):
      - Move the pickle loading and metadata cleanup from HW08 cell #4 here.
      - Build idx_to_class / class_to_idx mappings.
      - Apply the transform passed in by the caller.
    """

    def __init__(self, data_path: str | Path, transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.samples = []           # filled in during port
        self.idx_to_class = {}      # filled in during port

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raise NotImplementedError("Port from HW08 cell #4")

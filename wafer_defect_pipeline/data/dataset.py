"""Class-conditional WM-811K wafer map dataset.

Ported from HW08_20231049.ipynb (ConditionalWaferDataset).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_IMG_SIZE = 28


class ConditionalWaferDataset(Dataset):
    """Wafer map dataset that yields (image, class_index) pairs.

    Drops the 'none' (normal) class so the dataset contains defect patterns only.
    Wafer maps are resized with nearest-neighbor interpolation to preserve the
    original {0, 1, 2} pixel values, then mapped to grayscale and passed through
    the caller-supplied transform.
    """

    def __init__(
        self,
        data_path: str | Path,
        transform=None,
        img_size: int = DEFAULT_IMG_SIZE,
    ):
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"'{data_path}' 파일을 찾을 수 없습니다.")

        df = pd.read_pickle(data_path).copy()

        df["failureType"] = df["failureType"].apply(
            lambda x: x[0][0] if len(x) > 0 and isinstance(x, np.ndarray) else x
        )
        df = df[(df["failureType"] != "none") & (df["failureType"].notnull())]
        df.reset_index(drop=True, inplace=True)

        self.data = df
        self.transform = transform
        self.img_size = img_size

        self.classes = sorted(df["failureType"].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wafer_map = self.data.iloc[idx]["waferMap"]
        wafer_map = cv2.resize(
            wafer_map,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST,
        )
        img_array = (wafer_map / 2.0 * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        label_name = self.data.iloc[idx]["failureType"]
        label_idx = self.class_to_idx[label_name]
        return img, torch.tensor(label_idx, dtype=torch.long)

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, WeightedRandomSampler

from wafer_defect_pipeline.data import (
    ConditionalWaferDataset,
    build_transform,
    download_wm811k,
)
from wafer_defect_pipeline.models import ConsistencyModel, MyDDPM, MyUNet


def build_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def resolve_data_path(cfg: DictConfig) -> Path:
    source = cfg.data.dataset.source
    if source == "kagglehub":
        return download_wm811k(cache_dir=cfg.data.dataset.cache_dir)
    if source == "local_pickle":
        path = Path(cfg.data.dataset.pickle_path)
        if not path.exists():
            raise FileNotFoundError(f"local_pickle path not found: {path}")
        return path
    raise ValueError(f"Unknown data.dataset.source: {source}")


def build_dataset(cfg: DictConfig) -> ConditionalWaferDataset:
    path = resolve_data_path(cfg)
    transform = build_transform(img_size=cfg.data.image.size)
    return ConditionalWaferDataset(path, transform=transform, img_size=cfg.data.image.size)


def build_dataloader(cfg: DictConfig, dataset: ConditionalWaferDataset) -> DataLoader:
    if cfg.data.dataloader.weighted_sampler:
        labels = [int(dataset[i][1]) for i in range(len(dataset))]
        counts = torch.bincount(torch.tensor(labels))
        class_weights = (counts.sum() / counts.float()).tolist()
        sample_weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        return DataLoader(
            dataset,
            batch_size=cfg.data.dataloader.batch_size,
            sampler=sampler,
            num_workers=cfg.data.dataloader.num_workers,
            pin_memory=True,
        )
    return DataLoader(
        dataset,
        batch_size=cfg.data.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.data.dataloader.num_workers,
        pin_memory=True,
    )


def build_unet(cfg: DictConfig) -> MyUNet:
    return MyUNet(
        n_steps=cfg.model.n_steps,
        time_emb_dim=cfg.model.unet.time_emb_dim,
        num_classes=cfg.model.unet.num_classes,
    )


def build_ddpm(cfg: DictConfig, device: torch.device) -> MyDDPM:
    net = build_unet(cfg)
    return MyDDPM(
        net,
        n_steps=cfg.model.n_steps,
        min_beta=cfg.model.min_beta,
        max_beta=cfg.model.max_beta,
        device=device,
        image_chw=(cfg.data.image.channels, cfg.data.image.size, cfg.data.image.size),
    )


def build_cm(cfg: DictConfig, device: torch.device) -> ConsistencyModel:
    net = build_unet(cfg)
    return ConsistencyModel(
        net,
        n_steps=cfg.model.n_steps,
        sigma_min=cfg.model.sigma_min,
        sigma_max=cfg.model.sigma_max,
        device=device,
    )


def build_optimizer(cfg: DictConfig, params) -> torch.optim.Optimizer:
    name = cfg.train.optimizer.name.lower()
    lr = cfg.train.optimizer.lr
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")

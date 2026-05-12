from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from wafer_defect_pipeline.runtime import (
    build_dataloader,
    build_dataset,
    build_ddpm,
    build_device,
    build_optimizer,
)
from wafer_defect_pipeline.train import training_loop
from wafer_defect_pipeline.utils import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)
    device = build_device(cfg.train.device)
    print(f"device: {device}")

    dataset = build_dataset(cfg)
    loader = build_dataloader(cfg, dataset)
    print(f"dataset: {len(dataset)} samples, classes={dataset.idx_to_class}")

    ddpm = build_ddpm(cfg, device)
    optim = build_optimizer(cfg, ddpm.parameters())

    store_path = Path(cfg.train.store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    history = training_loop(
        ddpm,
        loader,
        n_epochs=cfg.train.n_epochs,
        optim=optim,
        device=device,
        store_path=store_path,
    )
    print(f"final loss: {history[-1]:.5f}, best checkpoint: {store_path}")


if __name__ == "__main__":
    main()

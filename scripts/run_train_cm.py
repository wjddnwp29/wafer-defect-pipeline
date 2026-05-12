from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from wafer_defect_pipeline.runtime import (
    build_dataloader,
    build_dataset,
    build_ddpm,
    build_device,
)
from wafer_defect_pipeline.train import train_consistency_model
from wafer_defect_pipeline.utils import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.model.name != "cm":
        raise ValueError(
            "run_train_cm.py requires model=cm. Pass `model=cm` on the command line."
        )
    teacher_checkpoint = cfg.model.get("teacher_checkpoint")
    if teacher_checkpoint is None:
        raise ValueError("model.teacher_checkpoint must be set (path to a trained DDPM .pt)")

    seed_everything(cfg.seed)
    device = build_device(cfg.train.device)

    teacher_ddpm = build_ddpm(cfg, device)
    teacher_state = torch.load(teacher_checkpoint, map_location=device)
    teacher_ddpm.load_state_dict(teacher_state)
    teacher_ddpm.eval()

    dataset = build_dataset(cfg)
    loader = build_dataloader(cfg, dataset)

    store_path = Path(cfg.train.store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    student_cm, history = train_consistency_model(
        teacher_ddpm,
        loader,
        n_epochs=cfg.train.n_epochs,
        device=device,
        lr=cfg.train.optimizer.lr,
        store_path=store_path,
    )
    print(f"final loss: {history[-1]:.5f}, student checkpoint: {store_path}")


if __name__ == "__main__":
    main()

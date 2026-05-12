from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from wafer_defect_pipeline.inference import (
    generate_cm_with_nfe,
    generate_ddim_with_nfe,
    generate_ddpm_with_nfe,
)
from wafer_defect_pipeline.runtime import build_cm, build_ddpm, build_device
from wafer_defect_pipeline.utils import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)
    device = build_device(cfg.train.device)

    checkpoint = cfg.get("checkpoint")
    if checkpoint is None:
        raise ValueError("checkpoint must be set (path to a trained model .pt)")

    n_samples = cfg.get("n_samples", 16)
    out_dir = Path(cfg.get("out_dir", "outputs/samples"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.model.name == "ddpm":
        ddpm = build_ddpm(cfg, device)
        ddpm.load_state_dict(torch.load(checkpoint, map_location=device))
        x, classes, nfe, elapsed = generate_ddpm_with_nfe(
            ddpm, n_samples=n_samples, device=device,
            c=cfg.data.image.channels, h=cfg.data.image.size, w=cfg.data.image.size,
        )
    elif cfg.model.name == "ddim":
        ddpm = build_ddpm(cfg, device)
        ddpm.load_state_dict(torch.load(checkpoint, map_location=device))
        x, classes, nfe, elapsed = generate_ddim_with_nfe(
            ddpm, n_samples=n_samples, device=device,
            ddim_steps=cfg.model.sampling_steps, eta=cfg.model.eta,
            c=cfg.data.image.channels, h=cfg.data.image.size, w=cfg.data.image.size,
        )
    elif cfg.model.name == "cm":
        cm = build_cm(cfg, device)
        cm.load_state_dict(torch.load(checkpoint, map_location=device))
        x, classes, nfe, elapsed = generate_cm_with_nfe(
            cm, n_samples=n_samples, device=device,
            num_steps=cfg.get("num_steps", 1),
            c=cfg.data.image.channels, h=cfg.data.image.size, w=cfg.data.image.size,
        )
    else:
        raise ValueError(f"Unknown model.name: {cfg.model.name}")

    out_path = out_dir / f"{cfg.model.name}_samples.pt"
    torch.save({"x": x.cpu(), "classes": classes.cpu()}, out_path)
    print(f"generated {n_samples} samples in {elapsed:.2f}s (NFE={nfe}) -> {out_path}")


if __name__ == "__main__":
    main()

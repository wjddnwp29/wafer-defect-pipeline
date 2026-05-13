from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
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

    mlflow.set_tracking_uri(cfg.train.logging.mlflow_uri)
    mlflow.set_experiment(cfg.train.logging.experiment_name)

    with mlflow.start_run(run_name=f"ddpm-{cfg.model.n_steps}step"):
        mlflow.log_params(
            {
                "model": cfg.model.name,
                "n_steps": cfg.model.n_steps,
                "min_beta": cfg.model.min_beta,
                "max_beta": cfg.model.max_beta,
                "time_emb_dim": cfg.model.unet.time_emb_dim,
                "num_classes": cfg.model.unet.num_classes,
                "img_size": cfg.data.image.size,
                "batch_size": cfg.data.dataloader.batch_size,
                "weighted_sampler": cfg.data.dataloader.weighted_sampler,
                "optimizer": cfg.train.optimizer.name,
                "lr": cfg.train.optimizer.lr,
                "n_epochs": cfg.train.n_epochs,
                "seed": cfg.seed,
            }
        )

        history = training_loop(
            ddpm,
            loader,
            n_epochs=cfg.train.n_epochs,
            optim=optim,
            device=device,
            store_path=store_path,
        )

        mlflow.pytorch.log_model(ddpm, artifact_path="model")

    print(f"final loss: {history[-1]:.5f}, best checkpoint: {store_path}")


if __name__ == "__main__":
    main()

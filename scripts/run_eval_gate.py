from __future__ import annotations

import hydra
import mlflow
import mlflow.pytorch
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.models import Inception_V3_Weights

from wafer_defect_pipeline.eval import (
    InceptionFeatureExtractor,
    evaluate_fid_gate_from_images,
)
from wafer_defect_pipeline.inference import (
    generate_cm_with_nfe,
    generate_ddim_with_nfe,
    generate_ddpm_with_nfe,
)
from wafer_defect_pipeline.registry import register_model, set_alias
from wafer_defect_pipeline.runtime import build_dataset, build_device
from wafer_defect_pipeline.utils import seed_everything


def _generate_samples(
    cfg: DictConfig,
    model: torch.nn.Module,
    n_samples: int,
    device: torch.device,
) -> torch.Tensor:
    c = cfg.data.image.channels
    h = cfg.data.image.size
    w = cfg.data.image.size

    if cfg.model.name == "ddpm":
        x, _, _, _ = generate_ddpm_with_nfe(
            model, n_samples=n_samples, device=device, c=c, h=h, w=w,
        )
    elif cfg.model.name == "ddim":
        x, _, _, _ = generate_ddim_with_nfe(
            model, n_samples=n_samples, device=device,
            ddim_steps=cfg.model.sampling_steps, eta=cfg.model.eta,
            c=c, h=h, w=w,
        )
    elif cfg.model.name == "cm":
        x, _, _, _ = generate_cm_with_nfe(
            model, n_samples=n_samples, device=device,
            num_steps=cfg.get("num_steps", 1),
            c=c, h=h, w=w,
        )
    else:
        raise ValueError(f"Unknown model.name: {cfg.model.name}")
    return x


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    run_id = cfg.get("run_id")
    if run_id is None:
        raise ValueError("run_id must be set (e.g. +run_id=<mlflow_run_id>)")

    seed_everything(cfg.eval.seed)
    device = build_device(cfg.train.device)

    mlflow.set_tracking_uri(cfg.train.logging.mlflow_uri)

    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    model.to(device).eval()

    dataset = build_dataset(cfg)
    n = min(cfg.eval.n_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:n]
    real_images = torch.stack([dataset[int(i)][0] for i in indices])

    fake_images = _generate_samples(cfg, model, n_samples=n, device=device)

    weights = Inception_V3_Weights.DEFAULT if cfg.eval.inception.pretrained else None
    extractor = InceptionFeatureExtractor(weights=weights).to(device).eval()

    result = evaluate_fid_gate_from_images(
        real_images=real_images,
        fake_images=fake_images,
        feature_extractor=extractor,
        device=device,
        threshold=cfg.eval.threshold,
        batch_size=cfg.eval.batch_size,
    )

    print(f"FID: {result.fid:.3f}, threshold: {result.threshold}, passed: {result.passed}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("eval_fid", result.fid)
        mlflow.log_metric("eval_threshold", result.threshold)
        mlflow.log_metric("eval_passed", int(result.passed))

    if not result.passed:
        print("gate failed - model not registered")
        raise SystemExit(1)

    version = register_model(
        model_uri=f"runs:/{run_id}/model",
        name=cfg.eval.registered_model_name,
    )
    set_alias(
        name=cfg.eval.registered_model_name,
        version=version.version,
        alias=cfg.eval.alias_on_pass,
    )
    print(
        f"registered {cfg.eval.registered_model_name} v{version.version} "
        f"as alias={cfg.eval.alias_on_pass}"
    )


if __name__ == "__main__":
    main()

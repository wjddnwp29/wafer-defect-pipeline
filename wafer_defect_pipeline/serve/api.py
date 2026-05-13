from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Response
from mlflow.tracking import MlflowClient
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from wafer_defect_pipeline.inference import (
    generate_cm_with_nfe,
    generate_ddim_with_nfe,
    generate_ddpm_with_nfe,
)
from wafer_defect_pipeline.registry import load_by_alias
from wafer_defect_pipeline.serve.metrics import (
    generate_images_total,
    generate_latency_seconds,
    generate_requests_total,
)
from wafer_defect_pipeline.serve.schemas import GenerateRequest, GenerateResponse

# 학습 시 ConditionalWaferDataset.idx_to_class 순서와 동일해야 함
WM811K_CLASS_TO_IDX: dict[str, int] = {
    "Center": 0,
    "Donut": 1,
    "Edge-Loc": 2,
    "Edge-Ring": 3,
    "Loc": 4,
    "Near-full": 5,
    "Random": 6,
    "Scratch": 7,
}

_MODEL_CACHE: dict[str, tuple[torch.nn.Module, str]] = {}


def _settings() -> dict[str, Any]:
    return {
        "model_name": os.environ.get("WAFER_MODEL_NAME", "wafer-defect-ddpm"),
        "alias": os.environ.get("WAFER_MODEL_ALIAS", "champion"),
        "device": os.environ.get("WAFER_DEVICE", "cpu"),
        "image_channels": int(os.environ.get("WAFER_IMAGE_CHANNELS", "1")),
        "image_size": int(os.environ.get("WAFER_IMAGE_SIZE", "28")),
    }


def _load_model(model_name: str, alias: str, device: str) -> tuple[torch.nn.Module, str]:
    key = f"{model_name}@{alias}"
    if key not in _MODEL_CACHE:
        model = load_by_alias(model_name, alias)
        model.to(device).eval()
        client = MlflowClient()
        version = client.get_model_version_by_alias(name=model_name, alias=alias)
        _MODEL_CACHE[key] = (model, version.version)
    return _MODEL_CACHE[key]


def _tensor_to_base64_png(img: torch.Tensor) -> str:
    arr = ((img + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    if arr.shape[0] == 1:
        pil = Image.fromarray(arr[0], mode="L")
    else:
        pil = Image.fromarray(arr.transpose(1, 2, 0))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _generate_samples(
    model: torch.nn.Module,
    req: GenerateRequest,
    classes: torch.Tensor,
    device: torch.device,
    c: int,
    h: int,
    w: int,
) -> torch.Tensor:
    if req.sampler == "ddpm":
        x, _, _, _ = generate_ddpm_with_nfe(
            model, n_samples=req.n, device=device, classes=classes, c=c, h=h, w=w,
        )
    elif req.sampler == "ddim":
        x, _, _, _ = generate_ddim_with_nfe(
            model, n_samples=req.n, device=device, classes=classes,
            ddim_steps=req.steps, eta=0.0, c=c, h=h, w=w,
        )
    else:
        x, _, _, _ = generate_cm_with_nfe(
            model, n_samples=req.n, device=device, classes=classes,
            num_steps=req.steps, c=c, h=h, w=w,
        )
    return x


app = FastAPI(title="wafer-defect-pipeline api")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if req.defect_class not in WM811K_CLASS_TO_IDX:
        raise HTTPException(
            status_code=400,
            detail=f"unknown defect_class: {req.defect_class}",
        )

    cfg = _settings()
    device = torch.device(cfg["device"])
    started = time.perf_counter()

    try:
        model, version = _load_model(cfg["model_name"], cfg["alias"], cfg["device"])

        if req.seed is not None:
            torch.manual_seed(req.seed)

        class_idx = WM811K_CLASS_TO_IDX[req.defect_class]
        classes = torch.full((req.n,), class_idx, dtype=torch.long, device=device)

        with torch.no_grad():
            x = _generate_samples(
                model, req, classes, device,
                cfg["image_channels"], cfg["image_size"], cfg["image_size"],
            )

        images_b64 = [_tensor_to_base64_png(x[i]) for i in range(x.shape[0])]
        elapsed = time.perf_counter() - started

        generate_requests_total.labels(
            defect_class=req.defect_class, sampler=req.sampler, status="success",
        ).inc()
        generate_latency_seconds.labels(
            defect_class=req.defect_class, sampler=req.sampler,
        ).observe(elapsed)
        generate_images_total.labels(defect_class=req.defect_class).inc(req.n)

        return GenerateResponse(
            images=images_b64,
            defect_class=req.defect_class,
            n=req.n,
            sampler=req.sampler,
            steps=req.steps,
            model_name=cfg["model_name"],
            model_version=version,
            latency_ms=elapsed * 1000.0,
        )
    except HTTPException:
        raise
    except Exception as exc:
        generate_requests_total.labels(
            defect_class=req.defect_class, sampler=req.sampler, status="error",
        ).inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

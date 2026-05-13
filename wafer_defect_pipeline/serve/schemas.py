from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Sampler = Literal["ddpm", "ddim", "cm"]


class GenerateRequest(BaseModel):
    defect_class: str
    n: int = Field(1, ge=1, le=64)
    sampler: Sampler = "ddim"
    steps: int = Field(50, ge=1, le=1000)
    seed: int | None = None


class GenerateResponse(BaseModel):
    images: list[str]
    defect_class: str
    n: int
    sampler: Sampler
    steps: int
    model_name: str
    model_version: str
    latency_ms: float

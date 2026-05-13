from __future__ import annotations

import pytest
from pydantic import ValidationError

from wafer_defect_pipeline.serve.schemas import GenerateRequest, GenerateResponse


def test_generate_request_defaults():
    req = GenerateRequest(defect_class="Donut")
    assert req.n == 1
    assert req.sampler == "ddim"
    assert req.steps == 50
    assert req.seed is None


def test_generate_request_full_fields():
    req = GenerateRequest(
        defect_class="Center",
        n=8,
        sampler="cm",
        steps=4,
        seed=42,
    )
    assert req.n == 8
    assert req.sampler == "cm"
    assert req.steps == 4
    assert req.seed == 42


def test_generate_request_rejects_n_out_of_range():
    with pytest.raises(ValidationError):
        GenerateRequest(defect_class="Donut", n=0)
    with pytest.raises(ValidationError):
        GenerateRequest(defect_class="Donut", n=65)


def test_generate_request_rejects_steps_out_of_range():
    with pytest.raises(ValidationError):
        GenerateRequest(defect_class="Donut", steps=0)
    with pytest.raises(ValidationError):
        GenerateRequest(defect_class="Donut", steps=1001)


def test_generate_request_rejects_unknown_sampler():
    with pytest.raises(ValidationError):
        GenerateRequest(defect_class="Donut", sampler="foo")


def test_generate_response_serializes():
    resp = GenerateResponse(
        images=["iVBORw0KGgo="],
        defect_class="Donut",
        n=1,
        sampler="ddim",
        steps=50,
        model_name="wafer-defect-ddpm",
        model_version="3",
        latency_ms=123.4,
    )
    dumped = resp.model_dump()
    assert dumped["model_version"] == "3"
    assert dumped["latency_ms"] == pytest.approx(123.4)
    assert dumped["images"] == ["iVBORw0KGgo="]

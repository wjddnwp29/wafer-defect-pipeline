from __future__ import annotations

import base64
import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from wafer_defect_pipeline.serve.api import app


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    from wafer_defect_pipeline.models import MyDDPM, MyUNet
    from wafer_defect_pipeline.serve import api as api_module

    net = MyUNet(n_steps=20, num_classes=8)
    ddpm = MyDDPM(net, n_steps=20, device=torch.device("cpu"), image_chw=(1, 28, 28))
    ddpm.eval()

    monkeypatch.setattr(api_module, "load_by_alias", lambda name, alias: ddpm)

    class _FakeVersion:
        version = "1"

    class _FakeClient:
        def get_model_version_by_alias(self, name, alias):
            return _FakeVersion()

    monkeypatch.setattr(api_module, "MlflowClient", _FakeClient)
    api_module._MODEL_CACHE.clear()
    yield ddpm
    api_module._MODEL_CACHE.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _counter_value(counter, **labels) -> float:
    for metric_family in counter.collect():
        for sample in metric_family.samples:
            if sample.name.endswith("_total") and sample.labels == labels:
                return sample.value
    return 0.0


def test_healthz_returns_ok(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_returns_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert b"wafer_generate_requests_total" in response.content


def test_generate_returns_images_and_metadata(client):
    response = client.post(
        "/generate",
        json={
            "defect_class": "Donut",
            "n": 2,
            "sampler": "ddim",
            "steps": 5,
            "seed": 42,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["defect_class"] == "Donut"
    assert body["n"] == 2
    assert body["sampler"] == "ddim"
    assert body["model_version"] == "1"
    assert body["latency_ms"] > 0
    assert len(body["images"]) == 2

    img_bytes = base64.b64decode(body["images"][0])
    img = Image.open(io.BytesIO(img_bytes))
    assert img.mode == "L"
    assert img.size == (28, 28)


def test_generate_rejects_unknown_class(client):
    response = client.post(
        "/generate",
        json={"defect_class": "NotARealClass", "n": 1},
    )
    assert response.status_code == 400
    assert "unknown defect_class" in response.json()["detail"]


def test_generate_validation_error_on_n_out_of_range(client):
    response = client.post(
        "/generate",
        json={"defect_class": "Donut", "n": 0},
    )
    assert response.status_code == 422


def test_generate_increments_success_counter(client):
    from wafer_defect_pipeline.serve.metrics import generate_requests_total

    labels = {"defect_class": "Center", "sampler": "ddim", "status": "success"}
    before = _counter_value(generate_requests_total, **labels)

    response = client.post(
        "/generate",
        json={
            "defect_class": "Center",
            "n": 1,
            "sampler": "ddim",
            "steps": 5,
            "seed": 0,
        },
    )
    assert response.status_code == 200

    after = _counter_value(generate_requests_total, **labels)
    assert after > before

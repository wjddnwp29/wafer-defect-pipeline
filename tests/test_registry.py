from __future__ import annotations

import mlflow
import mlflow.pytorch
import pytest
import torch
import torch.nn as nn
from mlflow.tracking import MlflowClient

from wafer_defect_pipeline.registry import load_by_alias, register_model, set_alias


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture
def mlflow_local(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri


def _log_pytorch_model() -> str:
    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(_TinyNet(), artifact_path="model")
        return run.info.run_id


def test_register_model_returns_version(mlflow_local):
    run_id = _log_pytorch_model()
    version = register_model(
        model_uri=f"runs:/{run_id}/model",
        name="wafer-test-register",
    )
    assert version.name == "wafer-test-register"
    assert int(version.version) >= 1


def test_set_alias_assigns_to_version(mlflow_local):
    run_id = _log_pytorch_model()
    name = "wafer-test-alias"
    version = register_model(model_uri=f"runs:/{run_id}/model", name=name)
    set_alias(name=name, version=version.version, alias="champion")

    client = MlflowClient()
    by_alias = client.get_model_version_by_alias(name=name, alias="champion")
    assert by_alias.version == version.version


def test_load_by_alias_returns_module(mlflow_local):
    run_id = _log_pytorch_model()
    name = "wafer-test-load"
    version = register_model(model_uri=f"runs:/{run_id}/model", name=name)
    set_alias(name=name, version=version.version, alias="champion")

    loaded = load_by_alias(name=name, alias="champion")
    output = loaded(torch.randn(1, 4))
    assert output.shape == (1, 2)

from __future__ import annotations

import mlflow
import mlflow.pytorch
import torch.nn as nn
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient


def register_model(model_uri: str, name: str) -> ModelVersion:
    return mlflow.register_model(model_uri=model_uri, name=name)


def set_alias(name: str, version: int | str, alias: str) -> None:
    client = MlflowClient()
    client.set_registered_model_alias(name=name, alias=alias, version=str(version))


def load_by_alias(name: str, alias: str = "champion") -> nn.Module:
    model_uri = f"models:/{name}@{alias}"
    return mlflow.pytorch.load_model(model_uri)

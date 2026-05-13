from __future__ import annotations

import os
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template
from mlflow.tracking import MlflowClient

app = Flask(__name__)


def _settings() -> dict[str, str | None]:
    return {
        "model_name": os.environ.get("WAFER_MODEL_NAME", "wafer-defect-ddpm"),
        "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI"),
    }


def _format_timestamp(ms: int | None) -> str:
    if ms is None:
        return "-"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.get("/")
def versions():
    cfg = _settings()
    client = MlflowClient(tracking_uri=cfg["tracking_uri"])
    raw_versions = client.search_model_versions(f"name='{cfg['model_name']}'")

    rows = [
        {
            "version": v.version,
            "aliases": list(getattr(v, "aliases", []) or []),
            "current_stage": v.current_stage,
            "creation_timestamp": _format_timestamp(v.creation_timestamp),
            "run_id": v.run_id,
        }
        for v in raw_versions
    ]
    return render_template(
        "versions.html",
        model_name=cfg["model_name"],
        versions=rows,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)

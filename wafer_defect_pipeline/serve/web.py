from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx
from flask import Flask, jsonify, render_template, request
from mlflow.tracking import MlflowClient

app = Flask(__name__)

WM811K_CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]

SAMPLERS = ["ddim", "ddpm", "cm"]


def _settings() -> dict[str, Any]:
    return {
        "model_name": os.environ.get("WAFER_MODEL_NAME", "wafer-defect-ddpm"),
        "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI"),
        "experiment_name": os.environ.get(
            "WAFER_EXPERIMENT_NAME", "wafer-defect-mlops"
        ),
        "max_runs": int(os.environ.get("WAFER_HISTORY_MAX_RUNS", "50")),
        "api_url": os.environ.get("WAFER_API_URL", "http://localhost:8000"),
        "api_timeout": float(os.environ.get("WAFER_API_TIMEOUT", "60")),
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


@app.get("/runs")
def runs():
    cfg = _settings()
    client = MlflowClient(tracking_uri=cfg["tracking_uri"])
    experiment = client.get_experiment_by_name(cfg["experiment_name"])

    rows: list[dict[str, Any]] = []
    if experiment is not None:
        run_objects = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=cfg["max_runs"],
        )
        for r in run_objects:
            rows.append(
                {
                    "run_id": r.info.run_id,
                    "run_name": r.info.run_name or "-",
                    "status": r.info.status,
                    "started_at": _format_timestamp(r.info.start_time),
                    "n_epochs": r.data.params.get("n_epochs", "-"),
                    "best_loss": r.data.metrics.get("best_loss"),
                    "eval_fid": r.data.metrics.get("eval_fid"),
                    "eval_passed": r.data.metrics.get("eval_passed"),
                }
            )

    return render_template(
        "runs.html",
        experiment_name=cfg["experiment_name"],
        runs=rows,
    )


@app.route("/generate", methods=["GET", "POST"])
def generate():
    if request.method == "GET":
        return render_template(
            "generate.html",
            classes=WM811K_CLASSES,
            samplers=SAMPLERS,
        )

    cfg = _settings()
    payload: dict[str, Any] = {
        "defect_class": request.form["defect_class"],
        "n": int(request.form["n"]),
        "sampler": request.form["sampler"],
        "steps": int(request.form["steps"]),
    }
    seed_str = request.form.get("seed", "").strip()
    if seed_str:
        payload["seed"] = int(seed_str)

    try:
        response = httpx.post(
            f"{cfg['api_url']}/generate",
            json=payload,
            timeout=cfg["api_timeout"],
        )
    except httpx.HTTPError as exc:
        return render_template(
            "generate.html",
            classes=WM811K_CLASSES,
            samplers=SAMPLERS,
            form=payload,
            error=f"API call failed: {exc}",
        )

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        return render_template(
            "generate.html",
            classes=WM811K_CLASSES,
            samplers=SAMPLERS,
            form=payload,
            error=f"API error ({response.status_code}): {detail}",
        )

    return render_template(
        "generate.html",
        classes=WM811K_CLASSES,
        samplers=SAMPLERS,
        form=payload,
        result=response.json(),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)

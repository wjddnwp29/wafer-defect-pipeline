from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mlflow
from hydra import compose, initialize_config_dir


def _latest_run_id(tracking_uri: str, experiment_name: str) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if len(runs) == 0:
        raise RuntimeError(f"no runs found in experiment '{experiment_name}'")
    return runs.iloc[0]["run_id"]


def main() -> None:
    config_dir = str((Path(__file__).resolve().parent.parent / "configs").as_posix())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")

    train_cmd = [sys.executable, "scripts/run_train_ddpm.py"]
    print(">>> Step 1/2: training")
    print(f"$ {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True)

    run_id = _latest_run_id(
        cfg.train.logging.mlflow_uri,
        cfg.train.logging.experiment_name,
    )
    print(f"latest run_id: {run_id}")

    eval_cmd = [
        sys.executable,
        "scripts/run_eval_gate.py",
        f"+run_id={run_id}",
    ]
    print(">>> Step 2/2: eval gate")
    print(f"$ {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()

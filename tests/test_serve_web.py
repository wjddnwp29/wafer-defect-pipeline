from __future__ import annotations

import pytest


class _FakeVersion:
    def __init__(
        self,
        version: str,
        aliases: list[str],
        current_stage: str,
        creation_timestamp: int,
        run_id: str,
    ):
        self.version = version
        self.aliases = aliases
        self.current_stage = current_stage
        self.creation_timestamp = creation_timestamp
        self.run_id = run_id


@pytest.fixture
def fake_versions(monkeypatch):
    from wafer_defect_pipeline.serve import web as web_module

    data = [
        _FakeVersion("1", ["champion"], "None", 1700000000000, "abc123runid"),
        _FakeVersion("2", ["challenger"], "None", 1700001000000, "def456runid"),
    ]

    class _FakeClient:
        def __init__(self, tracking_uri=None):
            pass

        def search_model_versions(self, filter_string):
            return data

    monkeypatch.setattr(web_module, "MlflowClient", _FakeClient)
    return data


@pytest.fixture
def fake_empty_versions(monkeypatch):
    from wafer_defect_pipeline.serve import web as web_module

    class _FakeClient:
        def __init__(self, tracking_uri=None):
            pass

        def search_model_versions(self, filter_string):
            return []

    monkeypatch.setattr(web_module, "MlflowClient", _FakeClient)


@pytest.fixture
def client():
    from wafer_defect_pipeline.serve.web import app

    app.config["TESTING"] = True
    return app.test_client()


def test_healthz_returns_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_versions_page_renders_rows(client, fake_versions):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "wafer-defect-ddpm" in html
    assert "champion" in html
    assert "challenger" in html
    assert "abc123runid" in html
    assert "def456runid" in html


def test_versions_page_handles_empty(client, fake_empty_versions):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "no versions registered yet" in html


class _FakeExperiment:
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id


class _FakeRunInfo:
    def __init__(self, run_id: str, run_name: str, status: str, start_time: int):
        self.run_id = run_id
        self.run_name = run_name
        self.status = status
        self.start_time = start_time


class _FakeRunData:
    def __init__(self, params: dict, metrics: dict):
        self.params = params
        self.metrics = metrics


class _FakeRun:
    def __init__(self, info: _FakeRunInfo, data: _FakeRunData):
        self.info = info
        self.data = data


@pytest.fixture
def fake_runs(monkeypatch):
    from wafer_defect_pipeline.serve import web as web_module

    runs_data = [
        _FakeRun(
            _FakeRunInfo("runabc12345", "ddpm-1000step", "FINISHED", 1700000000000),
            _FakeRunData(
                {"n_epochs": "30"},
                {"best_loss": 0.0123, "eval_fid": 42.5, "eval_passed": 1.0},
            ),
        ),
        _FakeRun(
            _FakeRunInfo("rundef67890", "ddpm-200step", "FINISHED", 1700001000000),
            _FakeRunData(
                {"n_epochs": "10"},
                {"best_loss": 0.0345, "eval_fid": 78.1, "eval_passed": 0.0},
            ),
        ),
    ]

    class _FakeClient:
        def __init__(self, tracking_uri=None):
            pass

        def get_experiment_by_name(self, name):
            return _FakeExperiment("exp1")

        def search_runs(self, experiment_ids, order_by, max_results):
            return runs_data

    monkeypatch.setattr(web_module, "MlflowClient", _FakeClient)
    return runs_data


@pytest.fixture
def fake_missing_experiment(monkeypatch):
    from wafer_defect_pipeline.serve import web as web_module

    class _FakeClient:
        def __init__(self, tracking_uri=None):
            pass

        def get_experiment_by_name(self, name):
            return None

    monkeypatch.setattr(web_module, "MlflowClient", _FakeClient)


def test_runs_page_renders_rows(client, fake_runs):
    resp = client.get("/runs")
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "ddpm-1000step" in html
    assert "ddpm-200step" in html
    assert "pass" in html
    assert "fail" in html
    assert "runabc12" in html


def test_runs_page_handles_missing_experiment(client, fake_missing_experiment):
    resp = client.get("/runs")
    assert resp.status_code == 200
    assert "no runs yet" in resp.data.decode("utf-8")

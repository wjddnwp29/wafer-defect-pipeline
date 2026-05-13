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

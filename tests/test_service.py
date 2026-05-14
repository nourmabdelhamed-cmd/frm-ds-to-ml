import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from ipcv.service import create_app

pytestmark = pytest.mark.service


def test_service_health_endpoint():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_service_lists_baseline_scenario():
    client = TestClient(create_app())

    response = client.get("/scenarios")

    assert response.status_code == 200
    assert "baseline" in response.json()["scenarios"]

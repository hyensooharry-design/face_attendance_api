import os

os.environ["DUMMY_MODE"] = "1"
os.environ["AUTO_DOWNLOAD_MODELS"] = "0"

from fastapi.testclient import TestClient

from api.main import app


def test_root_and_health():
    with TestClient(app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert root.json()["ok"] is True
        assert root.json()["dummy_mode"] is True

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True

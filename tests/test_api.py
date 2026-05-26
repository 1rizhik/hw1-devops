import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()


def test_predict_endpoint():
    payload = {"variance": 0.5, "skewness": 1.0, "curtosis": 0.8, "entropy": -0.2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "is_fake" in response.json()
    assert "class_name" in response.json()


def test_predict_invalid_data():
    payload = {"variance": "invalid", "skewness": 1.0, "curtosis": 0.8, "entropy": -0.2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200

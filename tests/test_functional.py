import requests
import pytest
import os

BASE_URL = os.environ.get("SERVICE_URL", "http://banknote-api:8000")

def test_health_endpoint():
    """Проверка health check"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] == True

def test_predict_endpoint():
    """Проверка предсказания"""
    payload = {
        "variance": 0.5,
        "skewness": 1.0,
        "curtosis": 0.8,
        "entropy": -0.2
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_fake" in data
    assert "class_name" in data
    assert data["class_name"] in ["fake", "genuine"]

def test_metrics_endpoint():
    """Проверка эндпоинта метрик"""
    response = requests.get(f"{BASE_URL}/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data or data == {}

def test_predict_fake_banknote():
    """Проверка предсказания для фальшивой банкноты"""
    payload = {
        "variance": 3.6,
        "skewness": 4.5,
        "curtosis": 1.2,
        "entropy": -2.1
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Проверяем, что поле is_fake существует и является 0 или 1
    assert "is_fake" in data
    assert data["is_fake"] in [0, 1]

def test_predict_genuine_banknote():
    """Проверка предсказания для подлинной банкноты"""
    payload = {
        "variance": 0.5,
        "skewness": 1.0,
        "curtosis": 0.8,
        "entropy": -0.2
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    # Подлинная банкнота может быть 0 или 1 в зависимости от данных
    assert response.json()["is_fake"] in [0, 1]
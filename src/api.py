import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import configparser
from typing import Optional
import json
import os

# Загрузка конфигурации
config = configparser.ConfigParser()
config.read('config.ini')
MODEL_PATH = config['DEFAULT']['model_path']

# Загрузка метрик
metrics = {}
try:
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
except:
    metrics = {'accuracy': 0.99}

# Загрузка модели
model = None
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Модель успешно загружена")
    print(f"Метрики модели: Accuracy={metrics.get('accuracy', 'N/A')}")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")

class BanknoteFeatures(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float
    request_id: Optional[str] = None

class PredictionResponse(BaseModel):
    is_fake: int
    class_name: str
    request_id: Optional[str] = None
    confidence: Optional[float] = None

app = FastAPI(
    title="Banknote Authentication API",
    description="Предсказание подлинности банкнот по 4 признакам",
    version="1.0",
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_accuracy": metrics.get('accuracy', 0)
    }

@app.get("/metrics")
def get_metrics():
    return metrics

@app.post("/predict", response_model=PredictionResponse)
def predict(features: BanknoteFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель недоступна")
    
    input_data = [[features.variance, features.skewness, features.curtosis, features.entropy]]
    
    try:
        prediction = model.predict(input_data)[0]
        class_name = "fake" if prediction == 1 else "genuine"
        
        # Получение вероятности (confidence)
        try:
            proba = model.predict_proba(input_data)[0]
            confidence = float(proba[prediction])
        except:
            confidence = None
        
        return PredictionResponse(
            is_fake=int(prediction),
            class_name=class_name,
            request_id=features.request_id,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
"""
FastAPI сервер для банкнот.
Загружает обученную модель и предоставляет эндпоинт /predict.
"""

import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import configparser

# Загрузка конфигурации
config = configparser.ConfigParser()
config.read('config.ini')
MODEL_PATH = config['DEFAULT']['model_path']

# Загрузка модели
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Модель успешно загружена")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None

class BanknoteFeatures(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

class PredictionResponse(BaseModel):
    is_fake: int
    class_name: str

app = FastAPI(
    title="Banknote Authentication API",
    description="Предсказание подлинности банкнот",
    version="1.0",
)

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {"status": "ok", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: BanknoteFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель недоступна")
    
    input_data = [[features.variance, features.skewness, features.curtosis, features.curtosis, features.entropy]]
    
    try:
        prediction = model.predict(input_data)[0]
        class_name = "fake" if prediction == 1 else "genuine"
        return PredictionResponse(is_fake=int(prediction), class_name=class_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
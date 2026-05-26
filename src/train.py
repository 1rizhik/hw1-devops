import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import configparser
import os
import json

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def load_data():
    df = pd.read_csv('data/BankNote_Authentication.csv')
    X = df.drop('class', axis=1)
    y = df['class']
    return X, y

def train_model():
    config = load_config()
    X, y = load_data()
    
    # Создание и обучение модели
    model = LogisticRegression(
        C=float(config['model']['C']),
        max_iter=int(config['model']['max_iter']),
        random_state=int(config['DEFAULT']['random_state'])
    )
    model.fit(X, y)
    
    # Предсказания
    y_pred = model.predict(X)
    
    # Метрики
    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred)),
        'recall': float(recall_score(y, y_pred)),
        'f1_score': float(f1_score(y, y_pred))
    }
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    model_path = config['DEFAULT']['model_path']
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Сохранение метрик
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    with open('models/accuracy.txt', 'w') as f:
        f.write(f"{metrics['accuracy']:.4f}")
    
    print(f"Model saved to {model_path}")
    print(f"Metrics saved to models/metrics.json")
    
    return model, metrics

if __name__ == "__main__":
    train_model()
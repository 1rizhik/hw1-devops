import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
import configparser
import os

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def load_data():
    # Загрузка подготовленных данных
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
        max_iter=int(config['model']['max_iter'])
    )
    model.fit(X, y)
    
    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    model_path = config['DEFAULT']['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    print(f"Accuracy: {model.score(X, y):.4f}")
    
    # Сохранение accuracy в файл
    with open('models/accuracy.txt', 'w') as f:
        f.write(f"{model.score(X, y):.4f}")

if __name__ == "__main__":
    train_model()
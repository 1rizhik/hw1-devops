import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import configparser
import os

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def prepare_data():
    config = load_config()
    data_path = config['DEFAULT']['data_path']
    
    # Загрузка данных
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Проверка на пропуски
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Разделение на признаки и целевую переменную
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=float(config['DEFAULT']['test_size']),
        random_state=int(config['DEFAULT']['random_state']),
        stratify=y
    )
    
    print(f"Train size: {len(X_train)} (Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)})")
    print(f"Test size: {len(X_test)} (Class 0: {sum(y_test==0)}, Class 1: {sum(y_test==1)})")
    
    # Сохранение подготовленных данных (опционально)
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    prepare_data()
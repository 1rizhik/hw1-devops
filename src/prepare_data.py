import pandas as pd
import os
from sklearn.model_selection import train_test_split
import configparser

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def prepare_data():
    config = load_config()
    data_path = config['DEFAULT']['data_path']
    
    # Загрузка данных
    df = pd.read_csv(data_path)
    
    # Разделение на признаки и целевую переменную
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=float(config['DEFAULT']['test_size']),
        random_state=int(config['DEFAULT']['random_state'])
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    prepare_data()
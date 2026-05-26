import pytest
import pickle

def test_model_exists():
    import os
    assert os.path.exists('models/model.pkl'), "Model file not found"

def test_model_loads():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    assert model is not None
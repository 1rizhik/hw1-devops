import pytest
import pickle
import os


def test_model_exists():
    assert os.path.exists("models/model.pkl"), "Model file not found"


def test_model_loads():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    assert model is not None


def test_model_predicts():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    test_input = [[0.5, 1.0, 0.8, -0.2]]
    prediction = model.predict(test_input)
    assert prediction[0] in [0, 1]

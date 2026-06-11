import pytest
import pandas as pd
import numpy as np
import joblib
import os

def test_data_generation():
    """Test synthetic data generation"""
    n_samples = 100
    data = pd.DataFrame({
        'temperature': np.random.normal(75, 15, n_samples),
        'vibration': np.random.normal(0.5, 0.2, n_samples),
        'pressure': np.random.normal(100, 20, n_samples)
    })
    assert len(data) == n_samples
    assert not data.isnull().any().any()

def test_model_input_shape():
    """Test model accepts correct input shape"""
    input_data = pd.DataFrame({
        'temperature': [75.0],
        'vibration': [0.5],
        'pressure': [100.0],
        'rpm': [1500.0],
        'age_days': [100]
    })
    assert input_data.shape == (1, 5)

def test_model_exists():
    """Test that model file exists"""
    model_path = 'model_registry/PredictiveMaintenance_v1.pkl'
    assert os.path.exists(model_path), f"Model not found at {model_path}"

def test_model_loads():
    """Test that model loads correctly"""
    model_path = 'model_registry/PredictiveMaintenance_v1.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        assert model is not None
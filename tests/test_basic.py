import pytest
import pandas as pd
import numpy as np
from src.core.data_loader import WeatherDataLoader
from src.forecasters.ensemble.forecaster import EnsembleForecaster

def test_data_loader_init():
    loader = WeatherDataLoader(locations={"Test": (0, 0)})
    assert "Test" in loader.LOCATIONS

def test_ensemble_prediction():
    data = pd.DataFrame({
        'year': np.arange(2010, 2024),
        'avg_tmax': np.random.uniform(20, 30, 14),
        'avg_tmin': np.random.uniform(10, 20, 14)
    })
    ensemble = EnsembleForecaster(weights=[0.5, 0.5])
    ensemble.train(data, 'avg_tmax')
    prediction = ensemble.predict(2026)
    assert isinstance(prediction, float)
    assert 15 < prediction < 35


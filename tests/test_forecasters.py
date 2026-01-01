import pytest
import pandas as pd
from src.forecasters.regression.forecaster import RegressionForecaster

def test_theilsen_forecaster():
    # Sample data
    data = pd.DataFrame({
        'year': [2020, 2021, 2022, 2023],
        'temp': [25.0, 25.5, 26.0, 26.5]
    })
    
    model = RegressionForecaster(model_type='theilsen')
    model.train(data, 'temp')
    prediction = model.predict(2024)
    
    assert 26.0 < prediction < 28.0  # Reasonable range
    assert isinstance(prediction, float)



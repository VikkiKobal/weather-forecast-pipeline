import numpy as np
import pandas as pd
from src.core.base_forecaster import WeatherForecaster
from src.forecasters.regression.forecaster import RegressionForecaster
from src.forecasters.prophet.forecaster import ProphetForecaster

class EnsembleForecaster(WeatherForecaster):
    """Combines Regression and Prophet models using weighted averaging."""
    def __init__(self, weights=None):
        self.models = {
            'regression': RegressionForecaster(model_type='theilsen'),
            'prophet': ProphetForecaster()
        }
        self.weights = weights if weights else [0.5, 0.5]

    def train(self, data: pd.DataFrame, target_col: str):
        for model in self.models.values():
            model.train(data, target_col)

    def predict(self, year: int) -> float:
        preds = [m.predict(year) for m in self.models.values()]
        return float(np.average(preds, weights=self.weights))


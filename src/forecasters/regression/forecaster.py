import pandas as pd
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from src.core.base_forecaster import WeatherForecaster

class RegressionForecaster(WeatherForecaster):
    """Robust regression for trend modeling."""
    def __init__(self, model_type='theilsen'):
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'theilsen':
            self.model = TheilSenRegressor(random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_type}")

    def train(self, data: pd.DataFrame, target_col: str):
        X = data[['year']].values
        y = data[target_col].values
        self.model.fit(X, y)

    def predict(self, year: int) -> float:
        return float(self.model.predict([[year]])[0])


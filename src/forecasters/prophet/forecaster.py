import pandas as pd
from prophet import Prophet
from src.core.base_forecaster import WeatherForecaster

class ProphetForecaster(WeatherForecaster):
    """Prophet wrapper for yearly aggregated series."""
    def __init__(self):
        self.model = None

    def train(self, data: pd.DataFrame, target_col: str):
        df = pd.DataFrame({
            'ds': pd.to_datetime(data['year'].astype(str) + '-08-01'),
            'y': data[target_col]
        })
        
        self.model = Prophet(
            yearly_seasonality=False, 
            weekly_seasonality=False, 
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        self.model.fit(df)

    def predict(self, year: int) -> float:
        future = pd.DataFrame({'ds': [pd.to_datetime(f"{year}-08-01")]})
        forecast = self.model.predict(future)
        return float(forecast['yhat'].iloc[0])

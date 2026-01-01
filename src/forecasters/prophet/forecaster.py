import os
import sys
import logging

# Silence Prophet and CmdStanPy completely
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# Also block cmdstanpy INFO logs from standard output
class FilterInfo(logging.Filter):
    def filter(self, record):
        return record.levelno > logging.INFO

logging.getLogger('cmdstanpy').addFilter(FilterInfo())

# Redirect stderr to suppress the "Importing plotly failed" message
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
try:
    from prophet import Prophet
finally:
    sys.stderr = stderr

import pandas as pd
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


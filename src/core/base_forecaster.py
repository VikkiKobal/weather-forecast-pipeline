from abc import ABC, abstractmethod
import pandas as pd

class WeatherForecaster(ABC):
    """Base class for weather forecasting models."""
    
    @abstractmethod
    def train(self, data: pd.DataFrame, target_col: str):
        pass
    
    @abstractmethod
    def predict(self, year: int) -> float:
        pass




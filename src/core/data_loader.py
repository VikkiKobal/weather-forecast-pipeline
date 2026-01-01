import pandas as pd
import requests
import time
from datetime import datetime

class WeatherDataLoader:
    """Handles multi-location data fetching to represent national average."""
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(self, locations: dict = None):
        self.LOCATIONS = locations if locations else {
            "Kyiv": (50.45, 30.52),
            "Kharkiv": (49.99, 36.23),
            "Odesa": (46.48, 30.72),
            "Lviv": (49.84, 24.03),
            "Dnipro": (48.46, 35.04)
        }

    def fetch_historical_data(self, start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
        """Fetches and averages daily data across major regional centers."""
        all_locations_data = []
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        if end_year >= datetime.now().year:
            end_date = current_date
        else:
            end_date = f"{end_year}-12-31"
        
        for city, (lat, lon) in self.LOCATIONS.items():
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    params = {
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": f"{start_year}-01-01",
                        "end_date": end_date,
                        "daily": ["temperature_2m_max", "temperature_2m_min"],
                        "timezone": "GMT"
                    }
                    
                    resp = requests.get(self.BASE_URL, params=params, timeout=30)
                    resp.raise_for_status()
                    
                    daily = resp.json()["daily"]
                    df = pd.DataFrame({
                        "date": pd.to_datetime(daily["time"]),
                        f"tmax_{city}": daily["temperature_2m_max"],
                        f"tmin_{city}": daily["temperature_2m_min"]
                    })
                    all_locations_data.append(df.set_index("date"))
                    print(f"  Loaded data for {city}")
                    
                    # Small delay between successful requests
                    time.sleep(1.5)
                    break # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 3
                        print(f"  Rate limit hit for {city}, retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"  Failed to load {city} after {max_retries} attempts: {e}")
        
        if not all_locations_data:
            raise ValueError("Failed to load data from any location")
        
        # Merge and calculate national average
        combined = pd.concat(all_locations_data, axis=1)
        
        final_df = pd.DataFrame(index=combined.index)
        final_df["tmax"] = combined[[c for c in combined.columns if "tmax" in c]].mean(axis=1)
        final_df["tmin"] = combined[[c for c in combined.columns if "tmin" in c]].mean(axis=1)
        
        return final_df.reset_index()

    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes statistical outliers using IQR method."""
        initial_count = len(df)
        df = df.dropna(subset=['tmax', 'tmin']).copy()
        
        for col in ['tmax', 'tmin']:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
            
        return df

    def process_august_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates daily data into yearly August averages with quality checks."""
        df = self.clean_and_validate_data(df)
        august_df = df[df['date'].dt.month == 8].copy()
        
        # Ensure sufficient data coverage
        year_counts = august_df.groupby(august_df['date'].dt.year).size()
        valid_years = year_counts[year_counts >= 20].index
        august_df = august_df[august_df['date'].dt.year.isin(valid_years)]
        
        yearly = august_df.groupby(august_df['date'].dt.year).agg({
            'tmax': 'mean',
            'tmin': 'mean'
        }).reset_index()
        
        yearly.columns = ['year', 'avg_tmax', 'avg_tmin']
        
        # Trend indicators
        yearly['rolling_tmax'] = yearly['avg_tmax'].rolling(5, min_periods=1).mean()
        yearly['rolling_tmin'] = yearly['avg_tmin'].rolling(5, min_periods=1).mean()
        
        return yearly

    def get_raw_daily_data(self, start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
        """Returns raw daily data."""
        df = self.fetch_historical_data(start_year, end_year)
        return df


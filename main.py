import numpy as np
import yaml
import warnings
import os
import logging
import pickle
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress technical logs from libraries
warnings.filterwarnings("ignore")
os.environ['CMDSTANPY_LOG_LEVEL'] = '30'

from src.core.data_loader import WeatherDataLoader
# ... (rest of imports)
from src.forecasters.regression.forecaster import RegressionForecaster
from src.forecasters.prophet.forecaster import ProphetForecaster
from src.forecasters.ensemble.forecaster import EnsembleForecaster
from src.utils.evaluation import run_backtesting
from src.utils.reporting import format_results_table

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    logger.info("Initializing Weather Pipeline...")
    
    # 1. Data Loading
    loader = WeatherDataLoader(locations=config['data']['locations'])
    raw_df = loader.get_raw_daily_data(
        start_year=config['data']['start_year'], 
        end_year=config['data']['end_year']
    )
    
    # Assert data quality before processing
    assert not raw_df.empty, "Fetched data is empty"
    assert 'tmax' in raw_df.columns and 'tmin' in raw_df.columns, "Missing temperature columns!"
    
    processed_df = loader.process_august_data(raw_df)
    assert len(processed_df) >= 5, "Insufficient historical data (need at least 5 years)"
    
    target_max = config['data']['target_max']
    target_min = config['data']['target_min']
    
    # 2. Model Evaluation
    test_years = config.get('validation', {}).get('test_years', [2024, 2025])
    logger.info(f"Evaluating models (Backtesting {test_years})...")
    reg_results_max = run_backtesting(processed_df, target_max, RegressionForecaster, test_years=test_years, model_type='theilsen')
    pro_results_max = run_backtesting(processed_df, target_max, ProphetForecaster, test_years=test_years)
    reg_results_min = run_backtesting(processed_df, target_min, RegressionForecaster, test_years=test_years, model_type='theilsen')
    pro_results_min = run_backtesting(processed_df, target_min, ProphetForecaster, test_years=test_years)

    metrics = {
        "max_reg_mae": float(np.mean([r['error'] for r in reg_results_max])),
        "max_pro_mae": float(np.mean([r['error'] for r in pro_results_max])),
        "min_reg_mae": float(np.mean([r['error'] for r in reg_results_min])),
        "min_pro_mae": float(np.mean([r['error'] for r in pro_results_min]))
    }
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df['timestamp'] = datetime.now().isoformat()
    metrics_df.to_csv("data/metrics/latest_metrics.csv", index=False)
    logger.info("Metrics exported to data/metrics/latest_metrics.csv")

    # 3. Generating Forecasts
    logger.info("Generating national forecast for August 2026...")
    w_max = [0.6, 0.4] if metrics["max_reg_mae"] < metrics["max_pro_mae"] else [0.4, 0.6]
    ens_max = EnsembleForecaster(weights=w_max)
    ens_max.train(processed_df, target_max)
    forecast_max = ens_max.predict(2026)
    
    # Save model using pickle
    with open("models/ensemble_max.pkl", "wb") as f:
        pickle.dump(ens_max, f)

    w_min = [0.6, 0.4] if metrics["min_reg_mae"] < metrics["min_pro_mae"] else [0.4, 0.6]
    ens_min = EnsembleForecaster(weights=w_min)
    ens_min.train(processed_df, target_min)
    forecast_min = ens_min.predict(2026)
    
    with open("models/ensemble_min.pkl", "wb") as f:
        pickle.dump(ens_min, f)
    
    logger.info("Models saved to models/ directory")

    # 4. Final Output
    print("\n" + "="*40)
    print(" NATIONAL WEATHER FORECAST: AUGUST 2026")
    print("="*40)
    print(f"  Average Maximum Temperature: {forecast_max:.2f} °C")
    print(f"  Average Minimum Temperature: {forecast_min:.2f} °C")
    print("="*40)
    
    val_results = {
        'Max': {'Regression': reg_results_max, 'Prophet': pro_results_max},
        'Min': {'Regression': reg_results_min, 'Prophet': pro_results_min}
    }
    
    forecasts_2026 = {
        'Ensemble': {'max': forecast_max, 'min': forecast_min}
    }
    
    table = format_results_table(processed_df, val_results, forecasts_2026)
    
    # Visualization: Print table to console using pandas
    print("\nDETAILED FORECAST REPORT")
    print("-" * 24)
    # Extract data from the markdown table format or recreate it for display
    # For simplicity and clear output, we'll print the summary table content
    print(table)
    print("-" * 40)
    
    with open("PREDICTIONS_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("# Weather Forecast: August 2026 (Ukraine)\n\n")
        f.write(f"### **Predicted Avg Max: {forecast_max:.2f} °C**\n")
        f.write(f"### **Predicted Avg Min: {forecast_min:.2f} °C**\n\n")
        f.write("## Detailed Results\n")
        f.write(table)
    
    logger.info("Summary report updated in PREDICTIONS_SUMMARY.md")

if __name__ == "__main__":
    main()

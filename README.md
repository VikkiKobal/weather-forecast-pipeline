# Weather Forecast: August 2026 (Ukraine)

This project is a Python pipeline for forecasting average temperatures in Ukraine for August 2026. It uses historical data from 2010 to 2025 to build an ensemble of models.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the pipeline:
   ```bash
   python main.py
   ```

## My Approach

### 1. Data Selection (2010 - 2025)
- I selected data starting from 2010 to focus on recent climate patterns. This avoids using older data that may not represent current temperature norms.

### 2. Model Selection (Ensemble)
I used an ensemble of two models to improve prediction stability:
- **Theil-Sen Estimator**: A robust regression model that is not affected by unusual weather years (outliers).
- **Facebook Prophet**: A time-series model designed to handle trends and seasonal changes effectively.
- **Selection Logic**: The final prediction uses a weighted average. The weights are determined by which model performed better during the validation phase (2023-2025).

## Project Structure

- `src/`: Contains data loading, model logic, and utilities.
- `config/`: Configuration files for locations and model parameters.
- `models/`: Folder for saved model artifacts (.pkl files).
- `logs/`: Folder for execution logs (pipeline.log).
- `data/metrics/`: Performance metrics saved in CSV format.
- `tests/`: Automated unit tests for code verification.

## Features

- **Model Registry**: Models are saved using pickle for reuse.
- **Experiment Tracking**: MAE metrics are exported to CSV for every execution.
- **Logging**: All steps are recorded in a log file for traceability.
- **Data Validation**: The script checks data quality before processing.
- **CI/CD**: Includes a GitHub Actions workflow for automated testing.

## Testing

Run tests with the following command:
```bash
python -m pytest tests/
```

## Data Source
Data provided by [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).

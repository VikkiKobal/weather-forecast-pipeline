import pandas as pd
import numpy as np

def run_backtesting(data: pd.DataFrame, target_col: str, model_class, test_years=None, **kwargs):
    """expanding window cross-validation for recent years."""
    results = []
    if test_years is None:
        test_years = [2024, 2025]
    
    available = data['year'].unique()
    valid_test_years = [y for y in test_years if y in available]
    
    for year in valid_test_years:
        train = data[data['year'] < year]
        actual = data[data['year'] == year][target_col].values[0]
        
        model = model_class(**kwargs)
        model.train(train, target_col)
        pred = model.predict(year)
        
        results.append({
            'year': year,
            'actual': actual,
            'predicted': pred,
            'error': abs(actual - pred)
        })
        
    return results


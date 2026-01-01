import pandas as pd

def format_results_table(history_data: pd.DataFrame, validation_results: dict, forecasts_2026: dict):
    """Generates comparative markdown table for validation and forecast years."""
    table_rows = []
    
    val_years = set()
    for target in validation_results:
        for model in validation_results[target]:
            for res in validation_results[target][model]:
                val_years.add(res['year'])
    
    history_years = history_data[~history_data['year'].isin(val_years)].tail(5)
    
    for _, row in history_years.iterrows():
        table_rows.append(f"| {int(row['year'])} | Historical | {row['avg_tmax']:.2f} | - | {row['avg_tmin']:.2f} | - | Actual |")

    models = list(validation_results['Max'].keys())
    years = sorted(list(val_years))
    
    for year in years:
        for model in models:
            res_max = next(r for r in validation_results['Max'][model] if r['year'] == year)
            res_min = next(r for r in validation_results['Min'][model] if r['year'] == year)
            
            table_rows.append(
                f"| {year} | {model} | "
                f"{res_max['actual']:.2f} | {res_max['predicted']:.2f} | "
                f"{res_min['actual']:.2f} | {res_min['predicted']:.2f} | Validated |"
            )

    for model, vals in forecasts_2026.items():
        table_rows.append(
            f"| 2026 | {model} | - | {vals['max']:.2f} | - | {vals['min']:.2f} | Forecast |"
        )

    header = "| Year | Model | Act Max | Pred Max | Act Min | Pred Min | Status |\n|------|-------|---------|----------|---------|----------|--------|"
    return header + "\n" + "\n".join(table_rows)


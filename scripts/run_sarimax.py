from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


# ============================================================
# IMPORT SARIMAX FUNCTIONS
# ============================================================

from src.sarimax_model import (
    grid_search,
    fit_selected,
    forecast,
    residual_diagnostics,
    calculate_metrics,
    plot_forecast,
)


# ============================================================
# DIRECTORIES
# ============================================================

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "appliance_hourly.csv"
)

METRICS_DIR = (
    ROOT
    / "outputs"
    / "metrics"
)

FORECAST_DIR = (
    ROOT
    / "outputs"
    / "forecasts"
)

FIGURES_DIR = (
    ROOT
    / "outputs"
    / "figures"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FORECAST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD HOURLY DATA
# ============================================================

print("\nLoading hourly appliance energy data...")

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"],
    index_col="date",
)

df = df.sort_index()


print(
    f"Dataset contains {len(df):,} hourly observations."
)


# ============================================================
# TARGET VARIABLE
# ============================================================

TARGET = "Appliances"

y = df[TARGET].astype(float)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

# Assignment uses a 24-hour forecasting horizon.
#
# We reserve the final 14 days as the evaluation period.
# The first 24 hours of this period are used for the
# main SARIMAX forecast evaluation.

TEST_DAYS = 14

TEST_HOURS = TEST_DAYS * 24

HORIZON = 24


train_end = len(df) - TEST_HOURS

y_train = y.iloc[:train_end]

y_test = y.iloc[train_end:]


# First 24 hours of the held-out period
y_test_24 = y_test.iloc[:HORIZON]


print("\nForecast design")
print("-----------------------------")

print(
    f"Training observations : {len(y_train):,}"
)

print(
    f"Held-out observations  : {len(y_test):,}"
)

print(
    f"Forecast horizon       : {HORIZON} hours"
)

print(
    f"Forecast start         : {y_test_24.index[0]}"
)

print(
    f"Forecast end           : {y_test_24.index[-1]}"
)


# ============================================================
# EXOGENOUS VARIABLES
# ============================================================

candidate_exog = [
    "T_out",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
]


exog_columns = [
    column
    for column in candidate_exog
    if column in df.columns
]


if exog_columns:

    X = df[exog_columns].copy()

    print(
        "\nExogenous variables:"
    )

    for column in exog_columns:
        print(f"  - {column}")

else:

    X = None

    print(
        "\nNo exogenous variables found."
    )


# Training exogenous variables
if X is not None:

    X_train = X.iloc[:train_end]

else:

    X_train = None


# ============================================================
# PART 4 — SARIMAX GRID SEARCH
# ============================================================

print("\n")
print("=" * 70)
print("SARIMAX GRID SEARCH")
print("=" * 70)

print(
    "\nSearching the required 147 models:"
)

print(
    "p = 0,...,6"
)

print(
    "d = 0,...,2"
)

print(
    "q = 0,...,6"
)

print(
    "Seasonal order = (1,1,1,24)"
)

print(
    "\nThis may take some time."
)


grid_results = grid_search(
    y=y_train,
    X=X_train,
    p_range=range(7),
    d_range=range(3),
    q_range=range(7),
    seasonal_order=(1, 1, 1, 24),
    maxiter=200,
)


# ============================================================
# SAVE GRID SEARCH RESULTS
# ============================================================

grid_results_path = (
    METRICS_DIR
    / "sarimax_grid_results.csv"
)

grid_results.to_csv(
    grid_results_path,
    index=False,
)


print(
    f"\nGrid search results saved to:"
)

print(
    grid_results_path
)


# ============================================================
# DISPLAY BEST MODELS
# ============================================================

successful_results = grid_results[
    grid_results["AIC"].notna()
].copy()


print("\n")
print("=" * 70)
print("TOP 10 SARIMAX MODELS BY AIC")
print("=" * 70)

print(
    successful_results[
        [
            "p",
            "d",
            "q",
            "P",
            "D",
            "Q",
            "s",
            "AIC",
            "BIC",
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_row = (
    successful_results
    .iloc[0]
)


best_order = (
    int(best_model_row["p"]),
    int(best_model_row["d"]),
    int(best_model_row["q"]),
)


seasonal_order = (
    int(best_model_row["P"]),
    int(best_model_row["D"]),
    int(best_model_row["Q"]),
    int(best_model_row["s"]),
)


print("\n")
print("=" * 70)
print("SELECTED SARIMAX MODEL")
print("=" * 70)

print(
    f"Order = {best_order}"
)

print(
    f"Seasonal order = {seasonal_order}"
)

print(
    f"AIC = {best_model_row['AIC']:.3f}"
)

print(
    f"BIC = {best_model_row['BIC']:.3f}"
)


# ============================================================
# FIT SELECTED MODEL
# ============================================================

print("\nFitting selected SARIMAX model...")

fit = fit_selected(
    y=y_train,
    order=best_order,
    seasonal_order=seasonal_order,
    X=X_train,
)


print(
    "\nSelected model fitted successfully."
)


# ============================================================
# MODEL SUMMARY
# ============================================================

summary_path = (
    METRICS_DIR
    / "sarimax_model_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        fit.summary().as_text()
    )


print(
    f"Model summary saved to:"
)

print(
    summary_path
)


# ============================================================
# RESIDUAL DIAGNOSTICS
# ============================================================

print("\n")
print("=" * 70)
print("RESIDUAL DIAGNOSTICS")
print("=" * 70)

ljung_box = residual_diagnostics(
    fit,
    FIGURES_DIR,
)


print(
    "\nLjung-Box test:"
)

print(
    ljung_box
)


# ============================================================
# 24-HOUR FORECAST
# ============================================================

print("\n")
print("=" * 70)
print("24-HOUR SARIMAX FORECAST")
print("=" * 70)


# IMPORTANT:
#
# Because the model contains exogenous weather variables,
# their values during the forecast period are supplied here.
#
# This is therefore a CONDITIONAL forecast using observed
# future covariates.
#
# This limitation must be discussed in the report.

if X is not None:

    X_forecast = X.iloc[
        train_end:
        train_end + HORIZON
    ]

else:

    X_forecast = None


forecast_df = forecast(
    fit=fit,
    horizon=HORIZON,
    index=y_test_24.index,
    X=X_forecast,
)


# Add actual values
forecast_df["actual"] = y_test_24.values


# ============================================================
# SAVE FORECAST
# ============================================================

forecast_path = (
    FORECAST_DIR
    / "sarimax_24h_forecast.csv"
)

forecast_df.to_csv(
    forecast_path
)


print(
    f"\nForecast saved to:"
)

print(
    forecast_path
)


print("\nForecast:")
print(
    forecast_df[
        [
            "actual",
            "forecast",
            "lower_95",
            "upper_95",
        ]
    ].to_string()
)


# ============================================================
# EVALUATION METRICS
# ============================================================

metrics = calculate_metrics(
    actual=y_test_24,
    predicted=forecast_df["forecast"],
)


metrics["AIC"] = fit.aic

metrics["BIC"] = fit.bic

metrics["p"] = best_order[0]
metrics["d"] = best_order[1]
metrics["q"] = best_order[2]

metrics["P"] = seasonal_order[0]
metrics["D"] = seasonal_order[1]
metrics["Q"] = seasonal_order[2]
metrics["seasonal_period"] = seasonal_order[3]


metrics_df = pd.DataFrame(
    [metrics]
)


metrics_path = (
    METRICS_DIR
    / "sarimax_24h_metrics.csv"
)


metrics_df.to_csv(
    metrics_path,
    index=False,
)


print("\n")
print("=" * 70)
print("SARIMAX PERFORMANCE")
print("=" * 70)

print(
    f"MAE   : {metrics['MAE']:.4f}"
)

print(
    f"RMSE  : {metrics['RMSE']:.4f}"
)

print(
    f"sMAPE : {metrics['sMAPE']:.4f}%"
)

print(
    f"Bias  : {metrics['Bias']:.4f}"
)


# ============================================================
# FORECAST PLOT
# ============================================================

forecast_plot_path = (
    FIGURES_DIR
    / "sarimax_24h_forecast.png"
)


plot_forecast(
    train=y_train,
    actual=y_test_24,
    forecast_df=forecast_df,
    output_path=forecast_plot_path,
)


print(
    "\nForecast plot saved to:"
)

print(
    forecast_plot_path
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("SARIMAX ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nGenerated files:"
)

print(
    f"1. {grid_results_path}"
)

print(
    f"2. {summary_path}"
)

print(
    f"3. {metrics_path}"
)

print(
    f"4. {forecast_path}"
)

print(
    f"5. {forecast_plot_path}"
)

print(
    f"6. {FIGURES_DIR / 'sarimax_residual_acf.png'}"
)

print(
    f"7. {FIGURES_DIR / 'sarimax_residual_distribution.png'}"
)

print(
    f"8. {FIGURES_DIR / 'sarimax_ljung_box.csv'}"
)

print("\nDone.")
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_download import download_raw_data
from src.preprocessing import run_preprocessing
from src.benchmarks import (
    mean_forecast,
    naive_forecast,
    seasonal_naive_forecast,
    drift_forecast,
)
from src.features import add_time_features, make_supervised_table, select_feature_columns
from src.ml_model import fit_model
from src.evaluation import evaluate


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_PATH = ROOT / "data" / "processed" / "appliance_hourly.csv"
OUTPUT_DIR = ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
FIGURE_DIR = OUTPUT_DIR / "figures"

for directory in [METRICS_DIR, FORECAST_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

if not DATA_PATH.exists():
    raw_path = download_raw_data()
    run_preprocessing(raw_path)

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"],
    index_col="date",
)

df = df.sort_index()

if not df.index.is_monotonic_increasing:
    raise ValueError("Datetime index is not sorted.")

if df.index.has_duplicates:
    raise ValueError("Duplicate timestamps detected.")

target = "Appliances"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found.")


# ---------------------------------------------------------------------
# Forecast design
# ---------------------------------------------------------------------

HORIZON = 24
TEST_DAYS = 14
TEST_STEPS = TEST_DAYS * 24

y = df[target]

train = y.iloc[:-TEST_STEPS]
test = y.iloc[-TEST_STEPS:]


# We evaluate the first 24 hours of the held-out 14-day period.
forecast_index = test.index[:HORIZON]


# ---------------------------------------------------------------------
# Benchmark forecasts
# ---------------------------------------------------------------------

forecasts = {}

forecasts["mean"] = mean_forecast(
    train,
    HORIZON,
    forecast_index,
)

forecasts["naive"] = naive_forecast(
    train,
    HORIZON,
    forecast_index,
)

forecasts["seasonal_naive_daily"] = seasonal_naive_forecast(
    train,
    HORIZON,
    forecast_index,
    seasonality=24,
)

forecasts["seasonal_naive_weekly"] = seasonal_naive_forecast(
    train,
    HORIZON,
    forecast_index,
    seasonality=168,
)

forecasts["drift"] = drift_forecast(
    train,
    HORIZON,
    forecast_index,
)


# ---------------------------------------------------------------------
# Leakage-safe feature model
# ---------------------------------------------------------------------
# This implementation performs recursive forecasting:
#
#   1. Fit using observations available before forecast origin.
#   2. Predict t+1.
#   3. Add the prediction to the history.
#   4. Construct features for t+2.
#   5. Continue until t+24.
#
# Therefore no actual target value from the forecast period is used
# as a lagged target feature.
# ---------------------------------------------------------------------


def prepare_training_table(history):
    """
    Build the supervised training table from historical observations.
    """

    table = make_supervised_table(history)

    if table.empty:
        raise ValueError(
            "Training table is empty. Not enough historical observations "
            "to construct lagged/rolling features."
        )

    return table


def build_future_features(history, timestamp, feature_columns):
    """
    Construct one feature row for a future timestamp.

    Only values contained in `history` are used for target-derived
    lag and rolling features.

    Time features are calculated directly from the timestamp.
    Other covariates are taken from the supplied dataframe when
    available.
    """

    row = pd.DataFrame(index=pd.DatetimeIndex([timestamp]))

    # -------------------------------------------------------------
    # Time features
    # -------------------------------------------------------------

    row["hour"] = timestamp.hour
    row["dayofweek"] = timestamp.dayofweek
    row["is_weekend"] = int(timestamp.dayofweek >= 5)

    row["hour_sin"] = np.sin(2 * np.pi * timestamp.hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * timestamp.hour / 24)

    row["dow_sin"] = np.sin(2 * np.pi * timestamp.dayofweek / 7)
    row["dow_cos"] = np.cos(2 * np.pi * timestamp.dayofweek / 7)

    # -------------------------------------------------------------
    # Target lags
    # -------------------------------------------------------------

    lag_values = [1, 2, 3, 6, 12, 24, 48, 72, 168]

    for lag in lag_values:
        lag_timestamp = timestamp - pd.Timedelta(hours=lag)

        if lag_timestamp not in history.index:
            raise ValueError(
                f"Missing historical value required for lag_{lag}: "
                f"{lag_timestamp}"
            )

        row[f"lag_{lag}"] = history.loc[lag_timestamp, target]

    # -------------------------------------------------------------
    # Rolling statistics
    # -------------------------------------------------------------

    rolling_windows = [3, 6, 12, 24, 168]

    for window in rolling_windows:

        end_timestamp = timestamp - pd.Timedelta(hours=1)
        start_timestamp = (
            end_timestamp - pd.Timedelta(hours=window - 1)
        )

        values = history.loc[start_timestamp:end_timestamp, target]

        if len(values) < window:
            raise ValueError(
                f"Insufficient history for rolling window {window} "
                f"at {timestamp}."
            )

        row[f"roll_mean_{window}"] = values.mean()
        row[f"roll_std_{window}"] = values.std()

    # -------------------------------------------------------------
    # Exogenous variables
    # -------------------------------------------------------------
    #
    # Future sensor/weather values are deliberately NOT used here.
    # This makes the primary ML result a genuine target forecast
    # rather than a conditional forecast using future observations.
    #
    # The time features and target-history features are therefore
    # available at the forecast origin/recursively during forecasting.
    # -------------------------------------------------------------

    missing = [
        c for c in feature_columns
        if c not in row.columns
    ]

    if missing:
        raise ValueError(
            "The following feature columns are unavailable for a "
            "leakage-safe forecast: "
            + ", ".join(missing)
        )

    return row[feature_columns]


# ---------------------------------------------------------------------
# Train feature model
# ---------------------------------------------------------------------

training_table = prepare_training_table(
    df.loc[df.index < forecast_index[0]].copy()
)

all_feature_columns = select_feature_columns(
    training_table,
    target=target,
)

# For the primary genuine-forecast experiment, use only features that
# can be generated without future sensor/weather observations.
allowed_feature_columns = [
    c
    for c in all_feature_columns
    if (
        c.startswith("lag_")
        or c.startswith("roll_")
        or c in [
            "hour",
            "dayofweek",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ]
    )
]

if not allowed_feature_columns:
    raise ValueError("No leakage-safe feature columns were selected.")

X_train = training_table[allowed_feature_columns]
y_train = training_table[target]

feature_model = fit_model(
    X_train,
    y_train,
)


# ---------------------------------------------------------------------
# Recursive 24-hour feature-model forecast
# ---------------------------------------------------------------------

history = df.loc[df.index < forecast_index[0], [target]].copy()

feature_predictions = []

for timestamp in forecast_index:

    X_future = build_future_features(
        history=history,
        timestamp=timestamp,
        feature_columns=allowed_feature_columns,
    )

    prediction = float(
        feature_model.predict(X_future)[0]
    )

    # Appliance usage cannot physically be negative.
    prediction = max(0.0, prediction)

    feature_predictions.append(prediction)

    # Add the prediction to the recursive history.
    history.loc[timestamp, target] = prediction

feature_forecast = pd.Series(
    feature_predictions,
    index=forecast_index,
    name="feature_model",
)

forecasts["feature_model"] = feature_forecast


# ---------------------------------------------------------------------
# Evaluate models
# ---------------------------------------------------------------------

actual = test.loc[forecast_index]

results = []

for model_name, prediction in forecasts.items():

    result = evaluate(
        model_name,
        actual,
        prediction,
        train,
    )

    results.append(result)


results_df = (
    pd.DataFrame(results)
    .sort_values("RMSE")
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------

metrics_path = (
    METRICS_DIR /
    "benchmark_feature_24h.csv"
)

results_df.to_csv(
    metrics_path,
    index=False,
)


# ---------------------------------------------------------------------
# Save forecasts
# ---------------------------------------------------------------------

forecast_df = pd.DataFrame(
    {
        "actual": actual,
        **forecasts,
    }
)

forecast_path = (
    FORECAST_DIR /
    "benchmark_feature_24h.csv"
)

forecast_df.to_csv(
    forecast_path
)


# ---------------------------------------------------------------------
# Save feature list
# ---------------------------------------------------------------------

feature_path = (
    METRICS_DIR /
    "feature_model_features.txt"
)

with open(feature_path, "w") as file:

    file.write(
        "Leakage-safe feature model\n"
        "==========================\n\n"
    )

    file.write(
        "Forecast horizon: 24 hours\n"
    )

    file.write(
        "Test period: final 14 days\n"
    )

    file.write(
        "Forecast type: recursive multi-step forecast\n\n"
    )

    file.write(
        "Features:\n"
    )

    for feature in allowed_feature_columns:
        file.write(f"- {feature}\n")


# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("BENCHMARK + LEAKAGE-SAFE FEATURE MODEL RESULTS")
print("=" * 70)

print(
    results_df.round(3).to_string(index=False)
)

print("\nMetrics saved to:")
print(metrics_path)

print("\nForecasts saved to:")
print(forecast_path)

print("\nFeature list saved to:")
print(feature_path)

print("\nForecast period:")
print(f"{forecast_index[0]} -> {forecast_index[-1]}")

print("\nDone.")
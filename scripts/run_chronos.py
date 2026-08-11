"""
Run the 24-hour Chronos foundation-model experiment.

The final 14 days are reserved as a held-out test period.
Only the first 24 hours of that test period are forecast, matching
the assignment requirement.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.chronos_model import (
    MODEL_ID,
    get_device,
    load_chronos_model,
    forecast_chronos,
)

from src.evaluation import evaluate


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "appliance_hourly.csv"
)

OUTPUT_DIR = ROOT / "outputs"

METRICS_DIR = OUTPUT_DIR / "metrics"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
FIGURE_DIR = OUTPUT_DIR / "figures"

for directory in [
    METRICS_DIR,
    FORECAST_DIR,
    FIGURE_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TARGET = "Appliances"

HORIZON = 24

TEST_DAYS = 14

TEST_STEPS = TEST_DAYS * 24

NUM_SAMPLES = 20


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"],
    index_col="date",
)

df = df.sort_index()

y = df[TARGET].astype(float)


# ---------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------

train = y.iloc[:-TEST_STEPS]

test = y.iloc[-TEST_STEPS:]

# Assignment requires a 24-hour forecast.
actual = test.iloc[:HORIZON]

forecast_index = actual.index


print()
print("=" * 70)
print("CHRONOS FOUNDATION MODEL")
print("=" * 70)

print(f"Model: {MODEL_ID}")
print(f"Training observations: {len(train)}")
print(f"Held-out period: {TEST_DAYS} days")
print(f"Forecast horizon: {HORIZON} hours")
print(f"Forecast origin: {forecast_index[0]}")
print(f"Forecast end: {forecast_index[-1]}")


# ---------------------------------------------------------------------
# Load Chronos
# ---------------------------------------------------------------------

pipeline, device = load_chronos_model(
    MODEL_ID
)


# ---------------------------------------------------------------------
# Generate forecast
# ---------------------------------------------------------------------

forecast, lower, upper = forecast_chronos(
    pipeline=pipeline,
    y_train=train,
    horizon=HORIZON,
    num_samples=NUM_SAMPLES,
)


# Ensure forecast index exactly matches test data.
forecast.index = forecast_index
lower.index = forecast_index
upper.index = forecast_index


# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------

metrics = evaluate(
    "chronos",
    actual,
    forecast,
    train,
)

metrics_df = pd.DataFrame(
    [metrics]
)


metrics_path = (
    METRICS_DIR
    / "chronos_24h_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)


# ---------------------------------------------------------------------
# Save forecast
# ---------------------------------------------------------------------

forecast_df = pd.DataFrame(
    {
        "actual": actual,
        "forecast": forecast,
        "lower_80": lower,
        "upper_80": upper,
    }
)


forecast_path = (
    FORECAST_DIR
    / "chronos_24h_forecast.csv"
)

forecast_df.to_csv(
    forecast_path
)


# ---------------------------------------------------------------------
# Plot forecast
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(14, 6)
)


# Seven days of history before forecast origin.
history = train.iloc[-7 * 24:]


ax.plot(
    history.index,
    history.values,
    color="steelblue",
    linewidth=1.5,
    label="Historical appliance use",
)


ax.plot(
    actual.index,
    actual.values,
    color="black",
    linewidth=2,
    label="Actual test data",
)


ax.plot(
    forecast.index,
    forecast.values,
    color="darkorange",
    linewidth=2,
    label="Chronos forecast",
)


ax.fill_between(
    forecast.index,
    lower.values,
    upper.values,
    color="darkorange",
    alpha=0.25,
    label="80% prediction interval",
)


ax.axvline(
    forecast_index[0],
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="Forecast origin",
)


ax.set_title(
    "Chronos 24-hour appliance-energy forecast"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Appliance energy use (Wh)"
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()


figure_path = (
    FIGURE_DIR
    / "chronos_24h_forecast.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Save experiment information
# ---------------------------------------------------------------------

info_path = (
    METRICS_DIR
    / "chronos_model_info.txt"
)


with open(
    info_path,
    "w",
) as f:

    f.write(
        "Chronos foundation-model experiment\n"
    )

    f.write(
        "===================================\n\n"
    )

    f.write(
        f"Model: {MODEL_ID}\n"
    )

    f.write(
        "Forecast type: zero-shot univariate\n"
    )

    f.write(
        f"Device: {device}\n"
    )

    f.write(
        f"Forecast horizon: {HORIZON} hours\n"
    )

    f.write(
        f"Held-out period: {TEST_DAYS} days\n"
    )

    f.write(
        f"Forecast samples: {NUM_SAMPLES}\n"
    )

    f.write(
        "Future covariates: none\n"
    )


# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------

print()
print("Chronos metrics:")
print(
    metrics_df.round(4).to_string(
        index=False
    )
)

print()
print("Generated files:")

print(
    metrics_path
)

print(
    forecast_path
)

print(
    figure_path
)

print(
    info_path
)

print()
print("Done.")
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

FORECAST_DIR = ROOT / "outputs" / "forecasts"
FIGURE_DIR = ROOT / "outputs" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load benchmark + feature-model forecasts
# ---------------------------------------------------------------------

benchmark_path = FORECAST_DIR / "benchmark_feature_24h.csv"

benchmark = pd.read_csv(
    benchmark_path,
    parse_dates=["date"],
    index_col="date",
)


# ---------------------------------------------------------------------
# Load SARIMAX forecast
# ---------------------------------------------------------------------

sarimax_path = FORECAST_DIR / "sarimax_24h_forecast.csv"

sarimax = pd.read_csv(
    sarimax_path,
    parse_dates=["date"],
    index_col="date",
)


# ---------------------------------------------------------------------
# Load Chronos forecast
# ---------------------------------------------------------------------

chronos_path = FORECAST_DIR / "chronos_24h_forecast.csv"

chronos = pd.read_csv(
    chronos_path,
    parse_dates=["date"],
    index_col="date",
)


# ---------------------------------------------------------------------
# Construct combined dataframe
# ---------------------------------------------------------------------

combined = pd.DataFrame(index=benchmark.index)

# Actual observations
combined["actual"] = benchmark["actual"]

# Benchmark models
benchmark_columns = [
    "mean",
    "naive",
    "seasonal_naive_daily",
    "seasonal_naive_weekly",
    "drift",
    "feature_model",
]

for column in benchmark_columns:
    if column in benchmark.columns:
        combined[column] = benchmark[column]


# SARIMAX
combined["sarimax"] = sarimax["forecast"].reindex(combined.index)

# Chronos
chronos_forecast_column = "chronos"

if chronos_forecast_column in chronos.columns:
    combined["chronos"] = chronos[chronos_forecast_column].reindex(
        combined.index
    )
else:
    # Fall back to the first non-actual numeric column if the
    # Chronos file uses a different prediction-column name.
    numeric_columns = chronos.select_dtypes("number").columns.tolist()

    numeric_columns = [
        c for c in numeric_columns
        if c not in ["actual", "lower_95", "upper_95"]
    ]

    if not numeric_columns:
        raise ValueError(
            "Could not identify the Chronos forecast column."
        )

    combined["chronos"] = chronos[numeric_columns[0]].reindex(
        combined.index
    )


# ---------------------------------------------------------------------
# Save combined forecast table
# ---------------------------------------------------------------------

combined_path = FORECAST_DIR / "final_model_comparison_24h.csv"

combined.to_csv(combined_path)

print(f"Combined forecast table saved to:")
print(combined_path)


# ---------------------------------------------------------------------
# Plot 1: all models
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(15, 8))

ax.plot(
    combined.index,
    combined["actual"],
    label="Actual",
    linewidth=3,
)

for column in [
    "mean",
    "naive",
    "seasonal_naive_daily",
    "seasonal_naive_weekly",
    "drift",
    "feature_model",
    "sarimax",
    "chronos",
]:
    if column in combined.columns:
        ax.plot(
            combined.index,
            combined[column],
            label=column.replace("_", " ").title(),
            linewidth=1.5,
        )


ax.set_title(
    "24-Hour Appliance Energy Forecast Comparison",
    fontsize=16,
)

ax.set_xlabel("Time")
ax.set_ylabel("Hourly Appliance Energy Use")

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
)

ax.grid(True, alpha=0.3)

fig.tight_layout()

all_models_path = FIGURE_DIR / "final_all_model_forecasts_24h.png"

fig.savefig(
    all_models_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("All-model forecast plot saved to:")
print(all_models_path)


# ---------------------------------------------------------------------
# Plot 2: strongest models
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(15, 8))

ax.plot(
    combined.index,
    combined["actual"],
    label="Actual",
    linewidth=3,
)

strong_models = [
    ("sarimax", "SARIMAX"),
    ("feature_model", "Feature-based ML"),
    ("chronos", "Chronos"),
    ("seasonal_naive_weekly", "Weekly Seasonal Naive"),
]

for column, label in strong_models:
    if column in combined.columns:
        ax.plot(
            combined.index,
            combined[column],
            label=label,
            linewidth=2,
        )


# Add SARIMAX confidence interval if available
if "lower_95" in sarimax.columns and "upper_95" in sarimax.columns:

    lower = sarimax["lower_95"].reindex(combined.index)
    upper = sarimax["upper_95"].reindex(combined.index)

    ax.fill_between(
        combined.index,
        lower,
        upper,
        alpha=0.15,
        label="SARIMAX 95% CI",
    )


ax.set_title(
    "Actual Appliance Energy Use and Strongest Forecasting Models",
    fontsize=16,
)

ax.set_xlabel("Time")
ax.set_ylabel("Hourly Appliance Energy Use")

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
)

ax.grid(True, alpha=0.3)

fig.tight_layout()

strong_models_path = (
    FIGURE_DIR / "final_strong_models_forecasts_24h.png"
)

fig.savefig(
    strong_models_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Strong-model forecast plot saved to:")
print(strong_models_path)


# ---------------------------------------------------------------------
# Final checks
# ---------------------------------------------------------------------

print("\nCombined forecast columns:")
print(combined.columns.tolist())

print("\nForecast period:")
print(combined.index.min())
print("to")
print(combined.index.max())

print("\nFinal forecast table:")
print(combined.round(2))
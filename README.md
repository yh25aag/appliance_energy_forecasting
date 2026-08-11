# Appliance Energy Forecasting

## Overview

This project investigates short-term electricity appliance-energy forecasting using the **Appliances Energy Prediction** dataset from the UCI Machine Learning Repository.

The aim is to model hourly appliance energy use and forecast the next **24 hours** using a range of increasingly complex forecasting approaches:

* Statistical benchmark forecasts
* Seasonal naive forecasts
* SARIMAX
* Feature-based machine learning
* Chronos time-series foundation model

The models are evaluated using a common held-out forecasting period and compared using MAE, RMSE, sMAPE, MASE and forecast bias.

The project was developed as part of a time-series forecasting assignment.

---

## Dataset

The analysis uses the UCI **Appliances Energy Prediction** dataset.

The original observations are recorded at 10-minute intervals. The dataset contains appliance energy consumption together with indoor temperature and humidity measurements, outdoor weather variables and timestamp information.

The target variable is:

```text
Appliances
```

representing appliance energy use in Wh.

The original dataset is available from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction

The dataset is described by Candanedo et al. (2017).

---

## Project Structure

```text
appliance_energy_forecasting/
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── appliance_hourly.csv
│
├── notebooks/
│   └── 01_appliance_energy_forecasting.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── 01_hourly_series.png
│   │   ├── 02_hour_of_day.png
│   │   ├── 03_day_of_week.png
│   │   ├── 04_acf.png
│   │   ├── 05_pacf.png
│   │   ├── 06_acf_difference.png
│   │   ├── sarimax_24h_forecast.png
│   │   ├── sarimax_residual_acf.png
│   │   ├── sarimax_residual_distribution.png
│   │   ├── chronos_24h_forecast.png
│   │   ├── final_all_model_forecasts_24h.png
│   │   └── final_strong_models_forecasts_24h.png
│   │
│   ├── forecasts/
│   │   ├── benchmark_feature_24h.csv
│   │   ├── sarimax_24h_forecast.csv
│   │   ├── chronos_24h_forecast.csv
│   │   └── final_model_comparison_24h.csv
│   │
│   └── metrics/
│       ├── benchmark_feature_24h.csv
│       ├── sarimax_24h_metrics.csv
│       ├── sarimax_grid_results.csv
│       ├── sarimax_model_summary.txt
│       ├── chronos_24h_metrics.csv
│       ├── chronos_model_info.txt
│       └── feature_model_features.txt
│
├── report/
│   └── README.md
│
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── run_eda.py
│   ├── run_pipeline.py
│   ├── run_sarimax.py
│   ├── run_chronos.py
│   └── make_final_forecast_plot.py
│
├── src/
│   ├── benchmarks.py
│   ├── chronos_model.py
│   ├── data_download.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── features.py
│   ├── ml_model.py
│   ├── preprocessing.py
│   └── sarimax_model.py
│
├── ASSIGNMENT_MAP.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methodology

### 1. Data preparation

The original 10-minute observations are parsed using the timestamp and resampled to hourly observations.

The preprocessing workflow:

1. Download the UCI dataset.
2. Parse the timestamp.
3. Sort observations chronologically.
4. Check the data for missing values.
5. Aggregate the 10-minute observations to hourly values.
6. Save the processed dataset to:

```text
data/processed/appliance_hourly.csv
```

---

### 2. Exploratory analysis

The exploratory analysis investigates:

* Overall appliance-energy behaviour
* Hour-of-day patterns
* Day-of-week patterns
* Autocorrelation
* Partial autocorrelation
* Differenced-series autocorrelation
* Stationarity

The generated figures are stored in:

```text
outputs/figures/
```

---

## Forecasting Design

The forecasting horizon is:

```text
24 hours
```

The final evaluation uses a chronological held-out period so that future observations are not used to train the models.

The analysis also uses a 14-day held-out period for the evaluation workflow while maintaining a 24-hour forecasting horizon.

The principal evaluation metrics are:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Symmetric Mean Absolute Percentage Error (sMAPE)
* Mean Absolute Scaled Error (MASE)
* Forecast bias

Lower values indicate better performance for MAE, RMSE, sMAPE and MASE. Bias is used to assess systematic over- or under-prediction.

---

## Models

### Benchmark models

The benchmark models are:

* Mean forecast
* Naive forecast
* Daily seasonal naive
* Weekly seasonal naive
* Drift forecast

These establish increasingly informative baselines before introducing more complex models.

### SARIMAX

A seasonal ARIMA model with exogenous variables was fitted using daily seasonality:

```text
Seasonal period = 24 hours
```

The non-seasonal parameters were selected using an AIC-based grid search over:

```text
p = 0,...,6
d = 0,...,2
q = 0,...,6
```

The selected model was:

```text
SARIMAX(5,0,6)(1,1,1,24)
```

with the selected weather variables included as exogenous predictors.

### Feature-based machine learning

A histogram gradient boosting regression model was used with:

* Time-of-day features
* Day-of-week features
* Weekend indicator
* Cyclical time features
* Lagged appliance demand
* Rolling means
* Rolling standard deviations
* Indoor/environmental measurements
* Weather variables

### Chronos

The foundation-model experiment uses:

```text
amazon/chronos-t5-small
```

as a zero-shot univariate time-series forecasting model.

No future covariates are supplied to Chronos.

---

## Results

The final 24-hour comparison was:

| Model                 |       MAE |      RMSE |      sMAPE |  MASE |     Bias |
| --------------------- | --------: | --------: | ---------: | ----: | -------: |
| **SARIMAX**           | **20.08** | **26.52** | **19.30%** |     — | **2.89** |
| Weekly seasonal naive |     30.00 |     48.81 |     23.19% | 0.562 |    -6.53 |
| Feature model         |     32.96 |     45.11 |     26.79% | 0.617 |    22.67 |
| Chronos               |     34.05 |     48.62 |     31.24% | 0.637 |   -32.10 |
| Mean                  |     39.38 |     50.01 |     37.61% | 0.737 |    -8.23 |
| Daily seasonal naive  |     80.28 |    118.04 |     44.20% | 1.503 |    58.89 |
| Naive                 |    242.64 |    247.60 |    110.21% | 4.542 |   242.64 |
| Drift                 |    243.88 |    248.76 |    110.47% | 4.565 |   243.88 |

SARIMAX achieved the lowest MAE and RMSE and substantially improved on the strongest benchmark, weekly seasonal naive.

The weekly seasonal naive model was the strongest benchmark, indicating that appliance energy use contains a meaningful weekly structure. However, the substantially better SARIMAX result indicates that repeating the previous week's profile alone does not fully capture the dynamics of appliance demand.

---

## SARIMAX Results

The selected model was:

```text
SARIMAX(5,0,6)(1,1,1,24)
```

with:

```text
AIC = 32158.343
BIC = 32277.786
```

The 24-hour forecast produced:

```text
MAE  = 20.083
RMSE = 26.524
sMAPE = 19.295%
Bias = 2.892
```

The model forecasts and confidence intervals are available in:

```text
outputs/forecasts/sarimax_24h_forecast.csv
```

and:

```text
outputs/figures/sarimax_24h_forecast.png
```

Residual diagnostics are also provided.

---

## Interpretation

The results suggest that appliance energy demand has strong temporal dependence and seasonal structure.

The weekly seasonal naive model substantially outperformed the simple mean, naive and drift approaches. This indicates that appliance demand is not adequately described by a constant level or a simple continuation of the most recent observation.

SARIMAX provided the strongest overall performance. Its advantage is consistent with the presence of daily seasonality, serial dependence and relationships with environmental variables.

The feature-based model also performed reasonably well, but its error remained above SARIMAX. This suggests that adding many predictors does not automatically result in improved forecasts. Feature engineering and nonlinear machine-learning flexibility are useful, but the statistical time-series structure remains highly informative for this dataset.

Chronos produced a competitive result but did not outperform either SARIMAX or the strongest seasonal benchmark. Therefore, the additional complexity of a foundation model is not justified solely by predictive accuracy in this experiment.

---

## Forecasting with Future Covariates

An important practical consideration is whether predictor variables are actually known at the forecast origin.

Past appliance demand and calendar variables such as:

* hour of day
* day of week
* weekend status

are available when the forecast is generated.

Future weather and indoor environmental variables are not necessarily known exactly. Using their realised future values from the test set would produce a conditional forecast rather than a genuine operational forecast.

The feature-based model therefore needs to be interpreted carefully if future environmental measurements are used during evaluation.

Chronos avoids this issue because it is a zero-shot univariate model and uses only the historical appliance-energy series.

---

## Final Recommendation

For this experiment, **SARIMAX is the recommended model**.

It achieved the lowest MAE and RMSE while also producing a relatively small forecast bias.

It provides a useful balance between:

* Forecast accuracy
* Interpretability
* Explicit modelling of seasonality
* Ability to incorporate exogenous variables
* Availability of confidence intervals
* Computational practicality
* Ease of deployment

The weekly seasonal naive model should nevertheless remain an important operational benchmark because it is extremely simple, transparent and inexpensive.

The foundation model is valuable as an experimental comparison, but its performance did not justify replacing the simpler statistical approach for this dataset and forecast horizon.

---

## Reproducibility

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download and prepare the data:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
```

Run exploratory analysis:

```bash
python scripts/run_eda.py
```

Run benchmark and feature-based models:

```bash
python scripts/run_pipeline.py
```

Run SARIMAX:

```bash
python scripts/run_sarimax.py
```

Run Chronos:

```bash
python scripts/run_chronos.py
```

Generate the final comparison:

```bash
python scripts/make_final_forecast_plot.py
```

---

## Key Output Files

The main final comparison is:

```text
outputs/metrics/benchmark_feature_24h.csv
outputs/metrics/sarimax_24h_metrics.csv
outputs/metrics/chronos_24h_metrics.csv
```

The combined forecast table is:

```text
outputs/forecasts/final_model_comparison_24h.csv
```

The main comparison figures are:

```text
outputs/figures/final_all_model_forecasts_24h.png
outputs/figures/final_strong_models_forecasts_24h.png
```

---

## References

Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings, 140*, 81–97.

Candanedo, L. M. (2017). Appliances Energy Prediction [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VC8G

The UCI dataset documentation is available at:

https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction

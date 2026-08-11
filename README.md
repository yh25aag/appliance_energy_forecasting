# Appliance Energy Demand Forecasting

GitHub-ready implementation of the Appliance Energy Prediction time-series assignment.

## Pipeline

1. Download raw UCI data: `python scripts/download_data.py`
2. Prepare hourly data: `python scripts/prepare_data.py`
3. Run EDA/stationarity: `python scripts/run_eda.py`
4. Run benchmarks + feature model: `python scripts/run_pipeline.py`
5. Run required 147 SARIMAX AIC search: `python scripts/run_sarimax.py`
6. Run Chronos-2: install its optional dependencies, then `python scripts/run_chronos.py`

## Repository structure

```text
├── data/raw/              # downloaded data (ignored by git)
├── data/processed/        # hourly data (ignored by git)
├── notebooks/             # analysis notebooks
├── src/                   # reusable implementation modules
├── scripts/               # runnable entry points
├── outputs/figures/       # generated figures
├── outputs/forecasts/     # generated forecasts
├── outputs/metrics/       # generated metrics
├── report/                # final report
├── requirements.txt
└── README.md
```

## Data extraction

The extraction is explicitly separated into `src/data_download.py` and `scripts/download_data.py`. The raw dataset is downloaded directly from the UCI URL supplied in the assignment. Preprocessing is handled by `src/preprocessing.py`: timestamp parsing, numeric conversion, missing-value reporting, hourly resampling and interpolation.

## Forecasting design

The assignment requires a 24-hour forecast horizon. Part 6 also specifies the final 14 days as the test period. The recommended evaluation is therefore rolling-origin 24-hour forecasting across that held-out 14-day period, rather than treating 14 days as one 336-step forecast.

## SARIMAX

`scripts/run_sarimax.py` implements all `p=0..6`, `d=0..2`, `q=0..6` combinations (147 models) with daily seasonal period 24 and ranks them by AIC.

## Chronos

Chronos is isolated in `src/chronos_model.py` because it has heavier dependencies. Do not fabricate Chronos results: run it in a compatible environment and copy the generated metrics into the final report.

## Reproducibility

The raw CSV and generated outputs are ignored by Git by default. The code downloads and regenerates them from the source dataset, making the repository reproducible without committing large generated files.

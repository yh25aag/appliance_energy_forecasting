from itertools import product
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def grid_search(
    y,
    X=None,
    p_range=range(7),
    d_range=range(3),
    q_range=range(7),
    seasonal_order=(1, 1, 1, 24),
    maxiter=200,
):
    """
    Search all required SARIMAX (p,d,q) combinations using AIC.

    Assignment requirement:
        p = 0,...,6
        d = 0,...,2
        q = 0,...,6

    This produces 7 x 3 x 7 = 147 combinations.

    Parameters
    ----------
    y : pd.Series
        Target time series.
    X : pd.DataFrame, optional
        Exogenous variables.
    p_range : range
        AR orders.
    d_range : range
        Differencing orders.
    q_range : range
        MA orders.
    seasonal_order : tuple
        Seasonal SARIMA order (P,D,Q,s).
    maxiter : int
        Maximum optimizer iterations.

    Returns
    -------
    pd.DataFrame
        Models ranked by AIC.
    """

    rows = []

    total_models = (
        len(list(p_range))
        * len(list(d_range))
        * len(list(q_range))
    )

    completed = 0

    for p, d, q in product(p_range, d_range, q_range):

        completed += 1

        print(
            f"Fitting model {completed}/{total_models}: "
            f"order=({p},{d},{q}), "
            f"seasonal_order={seasonal_order}"
        )

        try:
            model = SARIMAX(
                y,
                exog=X,
                order=(p, d, q),
                seasonal_order=seasonal_order,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            fit = model.fit(
                disp=False,
                maxiter=maxiter,
            )

            rows.append(
                {
                    "p": p,
                    "d": d,
                    "q": q,
                    "P": seasonal_order[0],
                    "D": seasonal_order[1],
                    "Q": seasonal_order[2],
                    "s": seasonal_order[3],
                    "AIC": fit.aic,
                    "BIC": fit.bic,
                    "converged": True,
                }
            )

        except Exception as error:

            rows.append(
                {
                    "p": p,
                    "d": d,
                    "q": q,
                    "P": seasonal_order[0],
                    "D": seasonal_order[1],
                    "Q": seasonal_order[2],
                    "s": seasonal_order[3],
                    "AIC": None,
                    "BIC": None,
                    "converged": False,
                    "error": str(error)[:200],
                }
            )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        "AIC",
        na_position="last",
    ).reset_index(drop=True)

    return results


def fit_selected(
    y,
    order,
    seasonal_order=(1, 1, 1, 24),
    X=None,
    maxiter=200,
):
    """
    Fit the selected SARIMAX model.
    """

    model = SARIMAX(
        y,
        exog=X,
        order=tuple(order),
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fit = model.fit(
        disp=False,
        maxiter=maxiter,
    )

    return fit


def forecast(
    fit,
    horizon,
    index,
    X=None,
):
    """
    Generate a forecast and 95% confidence intervals.
    """

    forecast_result = fit.get_forecast(
        steps=horizon,
        exog=X,
    )

    forecast_df = forecast_result.summary_frame(
        alpha=0.05
    )

    forecast_df = forecast_df.rename(
        columns={
            "mean": "forecast",
            "mean_se": "standard_error",
            "mean_ci_lower": "lower_95",
            "mean_ci_upper": "upper_95",
        }
    )

    forecast_df.index = index

    return forecast_df


def residual_diagnostics(
    fit,
    output_directory,
):
    """
    Produce SARIMAX residual diagnostics:

    1. Residual ACF
    2. Residual distribution
    3. Ljung-Box test
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    residuals = fit.resid.dropna()

    # ---------------------------------------------------------
    # Residual ACF
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    plot_acf(
        residuals,
        lags=168,
        ax=ax,
    )

    ax.set_title(
        "SARIMAX Residual Autocorrelation"
    )

    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")

    fig.tight_layout()

    fig.savefig(
        output_directory
        / "sarimax_residual_acf.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Residual distribution
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        residuals,
        bins=40,
    )

    ax.set_title(
        "SARIMAX Residual Distribution"
    )

    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")

    fig.tight_layout()

    fig.savefig(
        output_directory
        / "sarimax_residual_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Ljung-Box test
    # ---------------------------------------------------------

    ljung_box = acorr_ljungbox(
        residuals,
        lags=[24, 48],
        return_df=True,
    )

    ljung_box.to_csv(
        output_directory
        / "sarimax_ljung_box.csv"
    )

    return ljung_box


def calculate_metrics(
    actual,
    predicted,
):
    """
    Calculate common forecast accuracy metrics.

    Returns
    -------
    dict
        MAE, RMSE, sMAPE and Bias.
    """

    actual = pd.Series(actual)
    predicted = pd.Series(predicted)

    error = actual - predicted

    mae = error.abs().mean()

    rmse = (error ** 2).mean() ** 0.5

    denominator = (
        actual.abs() + predicted.abs()
    )

    smape = (
        100
        * (2 * error.abs() / denominator.replace(0, pd.NA))
    ).mean()

    bias = error.mean()

    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape,
        "Bias": bias,
    }


def plot_forecast(
    train,
    actual,
    forecast_df,
    output_path,
    title="SARIMAX 24-Hour Forecast",
):
    """
    Plot actual observations, forecast and
    95% confidence intervals.
    """

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    # Show only recent training data for readability
    recent_train = train.iloc[-7 * 24:]

    ax.plot(
        recent_train.index,
        recent_train.values,
        label="Training data",
    )

    ax.plot(
        actual.index,
        actual.values,
        label="Actual",
    )

    ax.plot(
        forecast_df.index,
        forecast_df["forecast"],
        label="SARIMAX forecast",
        linewidth=2,
    )

    ax.fill_between(
        forecast_df.index,
        forecast_df["lower_95"],
        forecast_df["upper_95"],
        alpha=0.2,
        label="95% confidence interval",
    )

    ax.set_title(title)

    ax.set_xlabel("Date")
    ax.set_ylabel("Appliance energy use")

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
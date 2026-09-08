"""
Post-processing analytics built entirely from existing model outputs and
forecasts. Nothing here retrains or fits a new model. Every function is
explicitly a labeled proxy/indicator, not a new predictive claim.
"""

import numpy as np
import pandas as pd
from typing import Optional


def cumulative_forecast(forecast_df: pd.DataFrame, horizons=(7, 14, 30)) -> dict:
    """
    Cumulative expected sale_amount over each horizon (in days). 7 days
    is the primary/default figure -- 14/30 are only meaningful if the
    forecast_df you pass in actually covers that many days (this simply
    sums however many forecast rows are available up to each horizon).
    """
    forecast_df = forecast_df.sort_values("dt")
    out = {}
    for h in horizons:
        out[f"cum_{h}d"] = float(forecast_df["forecast_sale_amount"].iloc[:h].sum())
    return out


def residual_uncertainty_band(test_actuals, test_preds, forecast_values, coverage: float = 0.8) -> pd.DataFrame:
    """
    EMPIRICAL RESIDUAL-BASED UNCERTAINTY BAND.

    This is explicitly NOT a confidence interval and NOT a guaranteed
    prediction interval -- those require distributional assumptions or a
    proper conformal-prediction procedure that this project doesn't
    implement. This band assumes the spread of residuals observed on the
    held-out test set is representative of future residual spread, which
    is a stated simplifying assumption, not a statistical guarantee.

    test_actuals / test_preds: arrays from your existing test-set evaluation
    (y_test, y_pred_best from the LightGBM notebook).
    forecast_values: array of forecasted sale_amount to attach a band to.
    """
    residuals = np.asarray(test_actuals) - np.asarray(test_preds)
    lower_q = (1 - coverage) / 2
    upper_q = 1 - lower_q
    lo_offset = np.quantile(residuals, lower_q)
    hi_offset = np.quantile(residuals, upper_q)

    forecast_values = np.asarray(forecast_values)
    return pd.DataFrame({
        "forecast": forecast_values,
        "band_low": np.maximum(forecast_values + lo_offset, 0),
        "band_high": forecast_values + hi_offset,
        "band_type": "empirical_residual_based",  # never rename this to "confidence_interval"
    })


def forecast_reliability_ranking(
    test_df: pd.DataFrame,
    actual_col: str = "sale_amount",
    pred_col: str = "prediction",
    group_cols=("store_id", "product_id"),
    top_n: int = 10,
) -> pd.DataFrame:
    """
    FORECAST RELIABILITY RANKING (formerly "risk ranking" -- renamed
    because "risk" implies a live operational signal this doesn't
    provide).

    Ranks store/product combinations by recent forecast error (MAE) and
    sales volatility (std of actuals) on the held-out test set. This
    identifies where the model has historically been LEAST RELIABLE /
    MOST VOLATILE -- it is NOT a claim about live business risk and NOT
    a stockout prediction.

    test_df needs columns: store_id, product_id, {actual_col}, {pred_col}.
    Build it once from your existing test-set predictions, e.g.:
        test_df = test[["store_id", "product_id", "sale_amount"]].copy()
        test_df["prediction"] = y_pred_best
        test_df.to_csv("test_predictions.csv", index=False)
    """
    df = test_df.copy()
    df["abs_error"] = (df[actual_col] - df[pred_col]).abs()

    grouped = df.groupby(list(group_cols)).agg(
        mean_abs_error=("abs_error", "mean"),
        sales_volatility=(actual_col, "std"),
        avg_sales=(actual_col, "mean"),
        n_obs=(actual_col, "count"),
    ).reset_index()

    grouped["forecast_error_score"] = (
        grouped["mean_abs_error"].rank(pct=True) * 0.5
        + grouped["sales_volatility"].rank(pct=True) * 0.5
    )
    # Higher forecast_error_score = historically less reliable / more volatile.

    return grouped.sort_values("forecast_error_score", ascending=False).head(top_n)

def store_product_performance(
    test_df: pd.DataFrame,
    store_id,
    product_id,
    actual_col: str = "sale_amount",
    pred_col: str = "prediction",
) -> Optional[dict]:
    """
    SELECTED STORE/PRODUCT ACCURACY.

    The dashboard's headline "LightGBM test R2 = 0.7096" is a GLOBAL
    average across every store/product pair. This function answers "how
    good is the model specifically for THIS pair" -- not a live risk
    signal, just a diagnostic.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    df = test_df.copy()
    pair = df[(df["store_id"] == store_id) & (df["product_id"] == product_id)]
    if pair.empty:
        return None

    mae = mean_absolute_error(pair[actual_col], pair[pred_col])
    rmse = float(np.sqrt(mean_squared_error(pair[actual_col], pair[pred_col])))
    r2 = r2_score(pair[actual_col], pair[pred_col]) if pair[actual_col].nunique() > 1 else None

    avg_sales = float(pair[actual_col].mean())
    normalized_error = (mae / avg_sales) if avg_sales > 0 else None

    df["abs_error"] = (df[actual_col] - df[pred_col]).abs()
    grouped = df.groupby(["store_id", "product_id"]).agg(
        mean_abs_error=("abs_error", "mean"),
        avg_sales=(actual_col, "mean"),
    ).reset_index()
    grouped = grouped[grouped["avg_sales"] > 0]
    grouped["normalized_error"] = grouped["mean_abs_error"] / grouped["avg_sales"]

    if normalized_error is None:
        reliability = "Unknown"
    elif normalized_error <= 0.30:
        reliability = "Good"
    elif normalized_error <= 0.50:
        reliability = "Moderate"
    else:
        reliability = "Poor"

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2) if r2 is not None else None,
        "avg_sales": avg_sales,
        "normalized_error": float(normalized_error) if normalized_error is not None else None,
        "percentile_rank": None,
        "reliability": reliability,
        "n_obs": int(len(pair)),
    }

def replenishment_indicator(inventory_assumption: float, lead_time_days: int, forecast_df: pd.DataFrame) -> dict:
    """
    FORECAST-BASED REPLENISHMENT INDICATOR (PROXY) -- never describe
    this as a true reorder point, actual inventory prediction, or actual
    stockout prediction.

    `inventory_assumption` and `lead_time_days` are USER-ENTERED
    ASSUMPTIONS, not measured data -- this dataset has no actual
    inventory quantity or confirmed lead-time field. The value is also
    deliberately NOT labeled "current inventory in sale_amount units":
    sale_amount is a sales figure, not a verified physical stock count,
    so calling it a unit of inventory would overstate what's actually
    known. Only use this if the person supplying `inventory_assumption`
    understands it must be expressed in the same units/scale as
    sale_amount for the comparison below to mean anything.

    Compares the user's inventory assumption against the model's
    forecasted sales over the user's lead-time assumption.
    """
    forecast_df = forecast_df.sort_values("dt")
    expected_lead_time_sales = float(forecast_df["forecast_sale_amount"].iloc[:lead_time_days].sum())
    at_risk = inventory_assumption < expected_lead_time_sales

    return {
        "expected_lead_time_sales": expected_lead_time_sales,
        "inventory_assumption": inventory_assumption,
        "lead_time_days": lead_time_days,
        "potential_stockout_risk": bool(at_risk),
        "label": "Forecast-Based Replenishment Indicator (Proxy) -- not a true reorder point or actual stockout prediction",
    }

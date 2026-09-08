"""
Recursive N-day sales forecasting using the trained LightGBM model.
7-day horizon is the primary/default use case.

FOLDER PLACEMENT
----------------
This file can live in any folder (e.g. `models/`, `notebooks/`, or
elsewhere) as long as:
  - best_lightgbm.pkl and lgbm_features.pkl sit in the SAME folder as
    this file (defaults below resolve relative to this file's own
    location, not the terminal's working directory).
  - the data folder (data/processed/sales_feature_engineered.parquet)
    is a SIBLING of whatever folder this file is in -- i.e. one level
    up from this script, then into data/processed/. That's what the
    __main__ smoke test at the bottom assumes.

WHAT THIS DOES
--------------
Turns the existing one-step-ahead LightGBM regressor into a genuine
multi-day-ahead forecaster by feeding each day's prediction back in as
the next day's lag_1 / lag_7 / etc. input.

DOCUMENTED ASSUMPTIONS -- read before trusting output past ~7 days
--------------------------------------------------------------------
1. Calendar features (year, month, day_of_month, day_of_week,
   week_of_year, quarter, is_weekend) are computed exactly from the
   future date. No assumption needed here -- these are genuinely known
   in advance.
2. lag_1/7/14/28 = the actual/forecast sale_amount from exactly that
   many days before the target date. rolling_mean_7/14/28 = mean of
   the TRAILING window ending the day before the target date (never
   including the target day itself -- the standard leakage-safe
   convention). All of these are recomputed fresh at every step from a
   running actual+forecast series, so they stay correct however far
   the recursion goes.
   NOTE: this convention has NOT been verified against the actual
   feature-engineering notebook that built these columns (that
   notebook wasn't reviewed). If the real definition differs, forecasts
   will be silently off rather than erroring -- confirm before fully
   trusting output past a quick sanity check.
3. holiday_flag: this dataset's holiday definition (which country/
   calendar it follows) isn't documented anywhere in the reviewed
   notebook, so it CANNOT be safely reproduced for future dates --
   guessing a calendar would be inventing data. `derive_holiday_flag()`
   below is a hook: if you know the exact rule used to build this
   column, implement it there and the pipeline switches to
   deterministic calculation automatically. Until then, holiday_flag is
   CARRIED FORWARD from the last known value, same as the other assumed
   columns below.
4. discount, activity_flag, and the weather block (precpt,
   avg_temperature, avg_humidity, avg_wind_level) are NOT knowable for
   future dates in this dataset. Default behaviour: CARRY FORWARD the
   last observed value per store/product. This is a stated simplifying
   assumption, not a prediction -- if you know a promotion or holiday
   is coming, pass it via `future_overrides`.
5. stock_hour6_22_cnt is excluded from the trained model (as it was
   during training) and is never forecasted or used here.
6. Categorical encoding (city_id/store_id/product_id): if you saved the
   exact training-time category dtypes (see `save_category_dtypes`
   below), those are loaded and used. Otherwise this falls back to
   reconstructing categories from the full historical dataframe's
   unique values -- the closest available approximation, not exact.

Forecast accuracy degrades the further out you go, especially on days
where the true discount/activity/holiday/weather value differs from the
carried-forward assumption. State this in the dashboard/README -- don't
let a smooth-looking forecast chart imply more certainty than this
actually supports.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from pandas.api.types import CategoricalDtype

CAT_COLS = ["city_id", "store_id", "product_id"]

CARRY_FORWARD_COLS = [
    "discount", "activity_flag", "holiday_flag",
    "precpt", "avg_temperature", "avg_humidity", "avg_wind_level",
]

# ASSUMPTION -- see module docstring point 2. Standard leakage-safe
# convention: lag_N = value N days back; rolling_mean_N = trailing N
# days ending the day BEFORE the target date. Not yet verified against
# the actual feature-engineering notebook.
LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]

REQUIRED_HISTORY_COLS = {"dt", "sale_amount", "store_id", "product_id", "city_id"}

# Resolve default artifact paths relative to THIS file's location, not
# whatever directory the script happens to be launched from (VS Code's
# Run button, some terminals, etc. don't guarantee cwd == script folder).
# This means best_lightgbm.pkl / lgbm_features.pkl just need to sit next
# to this script -- doesn't matter which folder that is.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "best_lightgbm.pkl")
DEFAULT_FEATURES_PATH = os.path.join(BASE_DIR, "lgbm_features.pkl")
DEFAULT_CAT_DTYPES_PATH = os.path.join(BASE_DIR, "lgbm_cat_dtypes.pkl")


def load_artifacts(model_path=DEFAULT_MODEL_PATH, features_path=DEFAULT_FEATURES_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature list artifact not found: {features_path}")
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

def get_lag_rolling_requirements(features):
    """Extract lag and rolling-window requirements from saved model features."""
    lag_days = []
    rolling_windows = []

    for feature in features:
        if feature.startswith("lag_"):
            lag_days.append(int(feature.split("_")[1]))
        elif feature.startswith("rolling_mean_"):
            rolling_windows.append(int(feature.split("_")[-1]))

    return sorted(lag_days), sorted(rolling_windows)


def save_category_dtypes(x_train: pd.DataFrame, cat_cols=CAT_COLS, path=DEFAULT_CAT_DTYPES_PATH):
    """
    Run this once from the training notebook, right after fitting the
    final model, to capture the EXACT category levels LightGBM was
    trained on:
        from forecast_pipeline import save_category_dtypes
        save_category_dtypes(x_train)
    """
    dtypes = {col: x_train[col].dtype for col in cat_cols if col in x_train.columns}
    joblib.dump(dtypes, path)
    return dtypes


def get_category_dtypes(history: pd.DataFrame, cat_cols=CAT_COLS, saved_path=DEFAULT_CAT_DTYPES_PATH) -> dict:
    """
    Prefer the exact training-time category dtypes if they were saved
    (see `save_category_dtypes`). Falls back to reconstructing category
    levels from the full historical dataframe's unique values -- a
    reasonable approximation, not guaranteed identical to training.
    """
    if os.path.exists(saved_path):
        try:
            return joblib.load(saved_path)
        except Exception:
            pass  # fall through to reconstruction rather than breaking the pipeline

    return {
        col: CategoricalDtype(categories=sorted(history[col].dropna().unique()))
        for col in cat_cols
    }


# Raw columns a NEW deployment-time data row must supply. These are the
# same raw/exogenous fields the training data has -- NOT lag/rolling
# columns and NOT calendar columns, both of which the pipeline always
# derives itself (calendar deterministically; lag/rolling from the
# combined actual+forecast series). Requiring the full raw set here
# (not just sale_amount) matches how CARRY_FORWARD_COLS is used: the
# newest row's discount/activity_flag/holiday_flag/weather values become
# the carry-forward basis for whatever gets forecast next.
NEW_DATA_REQUIRED_COLS = sorted(REQUIRED_HISTORY_COLS | set(CARRY_FORWARD_COLS))


def append_latest_data(history: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """
    DEPLOYMENT-PHASE helper: merges newly-arrived raw data (e.g. the most
    recent day or few days of actual sales) onto the existing historical
    dataframe, WITHOUT retraining or touching the saved model.

    new_data must contain one row per new day with these raw columns:
        dt, sale_amount, store_id, product_id, city_id,
        discount, activity_flag, holiday_flag,
        precpt, avg_temperature, avg_humidity, avg_wind_level
    (no lag_*, rolling_mean_*, or calendar columns -- those are always
    derived, never user-supplied.)

    If a (dt, store_id, product_id) in new_data already exists in
    history, the new_data row wins (treated as a correction/update).
    The result can be passed straight into `recursive_forecast` exactly
    like the original history dataframe -- no other changes needed,
    since lag/rolling features are always computed on the fly from
    sale_amount, not read from precomputed columns.
    """
    missing_cols = [c for c in NEW_DATA_REQUIRED_COLS if c not in new_data.columns]
    if missing_cols:
        raise ValueError(
            f"new_data is missing required column(s): {missing_cols}. "
            f"A deployment-time update needs the full raw feature set "
            f"(not just sale_amount), since discount/activity_flag/"
            f"holiday_flag/weather feed the carry-forward assumptions "
            f"used when forecasting forward."
        )

    new_data = new_data.copy()
    new_data["dt"] = pd.to_datetime(new_data["dt"])

    # Match dtypes to the existing history so concat/dedup/category
    # handling behaves consistently (e.g. store_id read from a CSV as
    # int64 vs. history's int64 -- keeps them aligned).
    for col in ("store_id", "product_id", "city_id"):
        if col in history.columns:
            try:
                new_data[col] = new_data[col].astype(history[col].dtype)
            except (ValueError, TypeError):
                pass  # leave as-is rather than failing merge on a benign dtype quirk

    combined = pd.concat([history, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=["dt", "store_id", "product_id"], keep="last")
    combined = combined.sort_values("dt").reset_index(drop=True)
    return combined


def derive_holiday_flag(target_date: pd.Timestamp) -> Optional[int]:
    """
    Hook for a deterministic holiday_flag calculation. Returns None by
    default, meaning "unknown -- fall back to carry-forward". If you
    know the exact calendar/country this dataset's holiday_flag follows,
    implement the rule here (e.g. using a fixed date list or the
    `holidays` package with the confirmed country code) and this will be
    used automatically instead of the carry-forward assumption.
    """
    return None


def predict_historical(
    hist_subset: pd.DataFrame,
    model,
    features: list,
    cat_dtypes: Optional[dict] = None,
) -> np.ndarray:
    """
    Runs the trained model's one-step-ahead prediction on already-
    engineered historical rows, using the REAL lag/rolling/calendar
    values that existed at the time -- not carried-forward assumptions.
    This is a model FIT/ACCURACY diagnostic (actual vs. predicted on
    history), distinct from `recursive_forecast`'s genuine future
    forecast -- don't label it as a forecast in the UI.

    hist_subset must already contain every column in `features`; this is
    true for the original feature-engineered dataframe (that's what the
    model was trained on). If some rows came from `append_latest_data`
    and lack precomputed lag/rolling values, LightGBM's native missing-
    value handling covers those rows -- no extra handling needed here.
    """
    missing_cols = [f for f in features if f not in hist_subset.columns]
    if missing_cols:
        raise ValueError(
            f"hist_subset is missing required feature column(s) {missing_cols}. "
            f"predict_historical expects the original feature-engineered "
            f"dataframe (with lag/rolling/calendar columns already computed), "
            f"not a raw/appended-only dataframe."
        )

    x = hist_subset.copy()
    cat_dtypes = cat_dtypes or {}
    for col in CAT_COLS:
        if col in x.columns and col in cat_dtypes:
            x[col] = x[col].astype(cat_dtypes[col])
    x = x[features]
    return model.predict(x)


def _calendar_features(dt: pd.Timestamp) -> dict:
    return {
        "year": dt.year,
        "month": dt.month,
        "day_of_month": dt.day,
        "day_of_week": dt.dayofweek,
        "week_of_year": int(dt.isocalendar().week),
        "quarter": dt.quarter,
        "is_weekend": int(dt.dayofweek >= 5),
    }


def _lag_and_rolling_features(sale_series: pd.Series, target_date: pd.Timestamp,lag_days: list,rolling_windows: list) -> dict:
    """
     Create lag and rolling-mean features using the requirements
     extracted from the saved model feature list.
    """
    feats = {}
    for d in lag_days:
        feats[f"lag_{d}"] = sale_series.get(target_date - pd.Timedelta(days=d), np.nan)

    for w in rolling_windows:
        window_start = target_date - pd.Timedelta(days=w)
        window_end = target_date - pd.Timedelta(days=1)
        window_vals = sale_series.loc[
            (sale_series.index >= window_start) & (sale_series.index <= window_end)
        ]
        feats[f"rolling_mean_{w}"] = float(window_vals.mean()) if not window_vals.empty else np.nan

    return feats

def _validate_history(
    history: pd.DataFrame,
    store_id,
    product_id,
    min_days: int = None,
    lag_days: list = None,
    rolling_windows: list = None
) -> pd.DataFrame:
    if min_days is None:
        lag_days = lag_days or LAG_DAYS
        rolling_windows = rolling_windows or ROLLING_WINDOWS
        min_days = max(lag_days + rolling_windows)

    missing_cols = REQUIRED_HISTORY_COLS - set(history.columns)
    if missing_cols:
        raise ValueError(f"History dataframe is missing required columns: {sorted(missing_cols)}")

    if store_id not in history["store_id"].unique():
        raise ValueError(f"Unknown store_id={store_id!r} -- not present in history.")

    if product_id not in history["product_id"].unique():
        raise ValueError(f"Unknown product_id={product_id!r} -- not present in history.")

    hist = (
        history[
            (history["store_id"] == store_id)
            & (history["product_id"] == product_id)
        ]
        .sort_values("dt")
        .copy()
    )

    if hist.empty:
        raise ValueError(
            f"No history rows found for store_id={store_id}, product_id={product_id}."
        )

    if len(hist) < min_days:
        raise ValueError(
            f"Only {len(hist)} day(s) of history for store_id={store_id}, product_id={product_id}; "
            f"need at least {min_days} to compute the longest lag/rolling feature "
            f"required by the saved model."
        )

    return hist

def recursive_forecast(
    history: pd.DataFrame,
    store_id,
    product_id,
    model,
    features: list,
    horizon: int = 7,
    cat_dtypes: Optional[dict] = None,
    future_overrides: Optional[dict] = None,
) -> pd.DataFrame:
    """
    history: full feature-engineered dataframe (needs at least 28 days
             of history for this store_id/product_id, to cover lag_28 /
             rolling_mean_28).
    horizon: forecast length in days. 7 is the intended default use case;
             longer horizons are supported but reliability degrades further.
    future_overrides: optional {column: {"YYYY-MM-DD": value}} to override
             a carry-forward assumption on a specific future date, e.g.
             {"discount": {"2025-02-15": 0.2}, "activity_flag": {"2025-02-15": 1}}.
    Returns one row per forecasted day, including a `forecast_sale_amount` column.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")

    future_overrides = future_overrides or {}
    lag_days, rolling_windows = get_lag_rolling_requirements(features)
    hist = _validate_history(history, store_id, product_id, lag_days=lag_days, rolling_windows=rolling_windows)
    cat_dtypes = cat_dtypes or get_category_dtypes(history)

    last_row = hist.iloc[-1]
    last_date = last_row["dt"]

    # running actual+forecast series, used to compute lag_1/7/14/28 and
    # rolling_mean_7/14/28 correctly at every step of the recursion,
    # including once we're beyond the window covered by actual data.
    sale_series = hist.set_index("dt")["sale_amount"].copy()

    carry_values = {col: last_row[col] for col in CARRY_FORWARD_COLS if col in hist.columns}
    static_values = {"city_id": last_row["city_id"], "store_id": store_id, "product_id": product_id}

    results = []

    for step in range(1, horizon + 1):
        target_date = last_date + pd.Timedelta(days=step)

        row = {}
        row.update(static_values)
        row.update(_calendar_features(target_date))
        row.update(carry_values)

        derived_holiday = derive_holiday_flag(target_date)
        if derived_holiday is not None:
            row["holiday_flag"] = derived_holiday
        # else: keep the carried-forward value already set above

        for col, overrides in future_overrides.items():
            key = target_date.strftime("%Y-%m-%d")
            if key in overrides:
                row[col] = overrides[key]

        row.update(_lag_and_rolling_features(sale_series, target_date, lag_days, rolling_windows))

        missing_features = [f for f in features if f not in row]
        if missing_features:
            raise ValueError(
                f"Cannot build forecast row for {target_date.date()}: "
                f"missing required feature(s) {missing_features}. "
                f"Check that `features` (from lgbm_features.pkl) matches the "
                f"columns this pipeline knows how to produce."
            )

        x = pd.DataFrame([row])
        for col in CAT_COLS:
            if col in x.columns:
                x[col] = x[col].astype(cat_dtypes[col])
        x = x[features]  # exact column set + order expected by the model

        pred = float(model.predict(x)[0])
        sale_series.loc[target_date] = pred

        row["dt"] = target_date
        row["forecast_sale_amount"] = pred
        results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Minimal smoke test. Paths are resolved relative to this file's
    # location (BASE_DIR), so this runs the same regardless of the
    # terminal's working directory or which folder this script sits in --
    # it just needs data/processed/ to be a sibling of that folder.
    model, features = load_artifacts()
    history_path = os.path.join(BASE_DIR, "..", "data", "processed", "sales_feature_engineered.parquet")
    history = pd.read_parquet(history_path)

    example_store = history["store_id"].iloc[0]
    example_product = history["product_id"].iloc[0]

    forecast = recursive_forecast(history, example_store, example_product, model, features, horizon=7)
    print(forecast[["dt", "forecast_sale_amount"]])

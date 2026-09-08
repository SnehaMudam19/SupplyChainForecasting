"""
Minimal Streamlit dashboard for the sales forecasting project.
7-day forecast is the default view.

FOLDER PLACEMENT
----------------
This file can sit in any folder (e.g. `models/`, `notebooks/`, or
elsewhere), as long as:
  - best_lightgbm.pkl, lgbm_features.pkl, forecast_pipeline.py,
    analytics.py, and (optionally) final_model_comparison.csv /
    test_predictions.csv sit in the SAME folder as this file.
  - data/processed/sales_feature_engineered.parquet is reachable one
    level up from that folder, at ../data/processed/ -- i.e. this
    script's folder and the data/ folder are siblings under the same
    project root. That's the layout this project already uses, so no
    changes needed if you keep that structure.

Run with:
    pip install streamlit plotly
    streamlit run dashboard.py

Every reliability/uncertainty/replenishment number shown here is
explicitly labeled as a model-derived proxy -- none of it is presented
as a measured business fact (actual stockout, true reorder point, or a
statistical confidence interval).
"""

import os

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from forecast_pipeline import (
    load_artifacts, recursive_forecast, get_category_dtypes,
    append_latest_data, NEW_DATA_REQUIRED_COLS, predict_historical,
)
from analytics import (
    cumulative_forecast,
    residual_uncertainty_band,
    forecast_reliability_ranking,
    replenishment_indicator,
    store_product_performance,
)

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# Resolve paths relative to THIS file's location, not whatever directory
# `streamlit run` happens to be launched from, and not tied to a
# specific folder name -- works from models/, notebooks/, or anywhere
# else at the same depth as data/.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_lightgbm.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "lgbm_features.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "sales_feature_engineered.parquet")
COMPARISON_PATH = os.path.join(BASE_DIR, "final_model_comparison.csv")
TEST_PREDICTIONS_PATH = os.path.join(BASE_DIR, "test_predictions.csv")


@st.cache_resource
def get_model():
    return load_artifacts(MODEL_PATH, FEATURES_PATH)


@st.cache_data
def get_history():
    return pd.read_parquet(HISTORY_PATH)


@st.cache_data
def get_test_predictions():
    if not os.path.exists(TEST_PREDICTIONS_PATH):
        return None
    try:
        df = pd.read_csv(TEST_PREDICTIONS_PATH)
        df["store_id"] = df["store_id"].astype(int)
        df["product_id"] = df["product_id"].astype(int)
        return df
    except Exception:
        return None

@st.cache_data
def get_lightgbm_r2():
    """
    Reads the LightGBM R2 from final_model_comparison.csv instead of
    hard-coding it, so the dashboard always shows whatever the notebook
    actually produced. Returns None if the file or the row can't be
    found, and the UI falls back to "N/A" rather than guessing.
    """
    if not os.path.exists(COMPARISON_PATH):
        return None
    try:
        comparison = pd.read_csv(COMPARISON_PATH)
    except Exception:
        return None

    # Check every column for a "lightgbm" mention regardless of dtype
    # (avoids a pandas FutureWarning from select_dtypes(include="object")
    # on newer pandas versions, and is just as correct here).
    mask = pd.Series(False, index=comparison.index)
    for col in comparison.columns:
        mask = mask | comparison[col].astype(str).str.contains("lightgbm", case=False, na=False)
    candidates = comparison[mask]

    r2_col = next((c for c in comparison.columns if c.strip().lower() in ("r2", "r²", "r_squared")), None)
    if candidates.empty or r2_col is None:
        return None
    return float(candidates.iloc[0][r2_col])


# ---------------- Load artifacts with visible error handling ----------------
missing = [p for p in (MODEL_PATH, FEATURES_PATH, HISTORY_PATH) if not os.path.exists(p)]
if missing:
    st.error(
        "Missing required file(s): " + ", ".join(missing) + ". "
        "Place best_lightgbm.pkl and lgbm_features.pkl in the same folder as "
        "this dashboard, and make sure data/processed/sales_feature_engineered.parquet "
        "is reachable one level up from that folder, then rerun."
    )
    st.stop()

try:
    model, features = get_model()
    history = get_history()
    cat_dtypes = get_category_dtypes(history)
except Exception as e:
    st.error(f"Failed to load model/history: {e}")
    st.stop()

# ---------------- Sidebar ----------------
st.sidebar.header("Filters")
store_id = st.sidebar.selectbox("Store", sorted(history["store_id"].unique()))

# Only offer products that actually exist for this store -- not every
# store sells every product, so a naive full product list lets the user
# pick a combination with zero history rows.
#available_products = sorted(history.loc[history["store_id"] == store_id, "product_id"].unique())
#
# product_id = st.sidebar.selectbox("Product", available_products)

test_predictions_df = get_test_predictions()
available_products = sorted(history.loc[history["store_id"] == store_id, "product_id"].unique())

if test_predictions_df is not None:
    testable_products = set(test_predictions_df.loc[test_predictions_df["store_id"] == store_id, "product_id"].unique())
    filtered = [p for p in available_products if p in testable_products]
    if filtered:
        available_products = filtered
    else:
        st.sidebar.caption("No products for this store have test-period data; showing all instead.")
else:
    st.sidebar.caption("test_predictions.csv not found -- accuracy sections will be unavailable.")

product_id = st.sidebar.selectbox("Product", available_products)

horizon_choice = st.sidebar.radio("Forecast horizon", ["Next day", "Next 7 days", "Custom"], index=1)
if horizon_choice == "Next day":
    horizon = 1
elif horizon_choice == "Next 7 days":
    horizon = 7
else:
    horizon = st.sidebar.slider("Custom horizon (days)", min_value=1, max_value=14, value=7)

st.sidebar.markdown("---")
st.sidebar.subheader("Deployment: add latest data")
st.sidebar.caption(
    "Add newly-arrived actual data WITHOUT retraining the model. Upload "
    "a CSV with one row per new day, using the same raw columns as the "
    "training data (no lag/rolling/calendar columns -- those are always "
    "computed automatically)."
)
st.sidebar.caption(f"Required columns: {', '.join(NEW_DATA_REQUIRED_COLS)}")

latest_data_file = st.sidebar.file_uploader("Latest data CSV", type=["csv"])
effective_history = history

if latest_data_file is not None:
    try:
        new_data = pd.read_csv(latest_data_file)
        effective_history = append_latest_data(history, new_data)
        st.sidebar.success(f"Merged {len(new_data)} new row(s). Forecasting will use the updated data.")
    except Exception as e:
        st.sidebar.error(f"Could not merge latest data: {e}")
        st.sidebar.caption("Forecasting will continue using the originally loaded data instead.")

st.sidebar.markdown("---")
st.sidebar.subheader("Optional: replenishment check")
st.sidebar.caption(
    "Both inputs below are your own assumptions, not measured data -- "
    "this dataset has no actual inventory or lead-time field."
)
inventory_assumption = st.sidebar.number_input(
    "Inventory assumption (same scale as sale_amount)", min_value=0.0, value=0.0
)
lead_time_days = st.sidebar.number_input("Lead time assumption (days)", min_value=1, max_value=horizon, value=min(3, horizon))
show_replenishment = st.sidebar.checkbox("Show replenishment indicator")

# ---------------- Main ----------------
st.title("Sales Forecast & Analytics")
st.caption(f"Store {store_id} · Product {product_id}")
if latest_data_file is not None and effective_history is not history:
    st.caption("Using historical data + your uploaded latest-data update (model was NOT retrained).")

hist_subset = (
    effective_history[(effective_history["store_id"] == store_id) & (effective_history["product_id"] == product_id)]
    .sort_values("dt")
    .tail(60)
)

try:
    effective_cat_dtypes = get_category_dtypes(effective_history) if effective_history is not history else cat_dtypes
    forecast_df = recursive_forecast(
        effective_history, store_id, product_id, model, features, horizon=horizon, cat_dtypes=effective_cat_dtypes
    )
except Exception as e:
    st.error(f"Could not generate forecast for this store/product: {e}")
    st.stop()

st.subheader("Recursive LightGBM Forecast")
st.caption(
    "This is the existing one-step LightGBM model rolled forward recursively, "
    "not a model trained to forecast multiple days directly."
)

show_actual_vs_predicted = st.checkbox(
    "Show model's historical predictions vs. actuals",
    value=False,
    help="Model fit diagnostic: how closely the model's one-step-ahead prediction "
         "tracked real sale_amount on past days. Not a forecast.",
)

# ---- Chart: actual history + (optional) historical model predictions + future forecast ----
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_subset["dt"], y=hist_subset["sale_amount"], mode="lines", name="Historical sale_amount (actual)"))

if show_actual_vs_predicted:
    try:
        hist_predictions = predict_historical(hist_subset, model, features, effective_cat_dtypes)
        fig.add_trace(go.Scatter(
            x=hist_subset["dt"], y=hist_predictions,
            mode="lines", name="Model prediction (historical, one-step-ahead)", line=dict(dash="dot"),
        ))
    except Exception as e:
        st.caption(f"Could not compute historical predictions: {e}")

fig.add_trace(go.Scatter(
    x=forecast_df["dt"], y=forecast_df["forecast_sale_amount"],
    mode="lines+markers", name=f"{horizon}-day recursive forecast (future)", line=dict(dash="dash"),
))
fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, width="stretch")

st.caption(
    "Limitation: discount, activity_flag, holiday_flag, and weather inputs "
    "beyond the last known day are carried forward from their last observed "
    "value (unless explicitly overridden), since future values for these "
    "aren't available in the dataset. Forecast reliability degrades the "
    "further out this recursion goes -- treat the 7-day view as primary."
)

col1, col2, col3 = st.columns(3)
r2 = get_lightgbm_r2()
col1.metric("LightGBM test R²", f"{r2:.4f}" if r2 is not None else "N/A")
if r2 is None:
    st.caption(f"({COMPARISON_PATH} not found or LightGBM R² row not identified -- showing N/A instead of a guessed value.)")

cum = cumulative_forecast(forecast_df, horizons=(min(7, horizon),))
col2.metric(f"Cumulative {list(cum.keys())[0]}", f"{list(cum.values())[0]:.2f}")
col3.metric("Forecast horizon", f"{horizon} days")

st.subheader(f"Selected Pair Accuracy — Store {store_id}, Product {product_id}")
st.caption(
    "The overall LightGBM R² above is a GLOBAL average across every store/product "
    "pair. This section shows how the model performed specifically for the pair "
    "you've selected -- this is why the actual-vs-predicted lines above can look "
    "quite different even when the overall R² looks solid."
)

if os.path.exists(TEST_PREDICTIONS_PATH):
    try:
        test_df = pd.read_csv(TEST_PREDICTIONS_PATH)
        perf = store_product_performance(test_df, store_id, product_id)
        if perf is None:
            st.info("No test-set rows found for this store/product pair.")
        else:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("MAE", f"{perf['mae']:.4f}")
            p2.metric("RMSE", f"{perf['rmse']:.4f}")
            p3.metric("R²", f"{perf['r2']:.4f}" if perf["r2"] is not None else "N/A (constant actuals)")
            p4.metric("Forecast Reliability", perf["reliability"])
            st.caption(
                f"Normalized error (MAE / avg sales): {perf['normalized_error']:.2%} "
                f"| Based on {perf['n_obs']} test-period observation(s) for this pair."
            )
    except Exception as e:
        st.info(f"Could not compute selected-pair accuracy: {e}")
else:
    st.info(f"{TEST_PREDICTIONS_PATH} not found yet -- see the steps above to generate it.")

if show_replenishment:
    st.subheader("Forecast-Based Replenishment Indicator (Proxy)")
    st.warning(
        "This is a PROXY built from your entered inventory and lead-time "
        "assumptions -- it is NOT actual stockout prediction and NOT a true "
        "reorder point."
    )
    result = replenishment_indicator(inventory_assumption, int(lead_time_days), forecast_df)
    c1, c2 = st.columns(2)
    c1.metric("Expected sales during lead time", f"{result['expected_lead_time_sales']:.2f}")
    c2.metric("Potential stockout risk (proxy)", "YES" if result["potential_stockout_risk"] else "NO")

st.subheader("Forecast Reliability Ranking")
st.caption(
    "Ranks store/product pairs by recent test-set forecast error and sales "
    "volatility -- where the model has historically been least reliable / "
    "most volatile. Not a live risk signal, not a stockout prediction."
)
if os.path.exists(TEST_PREDICTIONS_PATH):
    try:
        test_df = pd.read_csv(TEST_PREDICTIONS_PATH)
        ranking = forecast_reliability_ranking(test_df, top_n=10)
        st.dataframe(ranking, width="stretch")
    except Exception as e:
        st.info(f"Could not build the reliability ranking from {TEST_PREDICTIONS_PATH}: {e}")
else:
    st.info(
        f"{TEST_PREDICTIONS_PATH} not found. In your LightGBM notebook, after "
        "computing y_pred_best, add:\n\n"
        "test_out = test[['store_id','product_id','sale_amount']].copy()\n"
        "test_out['prediction'] = y_pred_best\n"
        f"test_out.to_csv('{TEST_PREDICTIONS_PATH}', index=False)"
    )

# Supply Chain Forecasting

Daily retail sales forecasting for store-product combinations, with a reproducible feature-engineering workflow, LightGBM and LSTM model comparison, and a Streamlit dashboard for forecasts and diagnostics.

## Project overview

The project predicts `sale_amount` from historical sales, calendar information, store/product identifiers, operational flags, discounts, and weather-related fields. LightGBM is the primary forecasting model; an LSTM is trained as a comparison model.

The repository includes exploratory analysis, validation utilities, feature-engineering notebooks, saved model artifacts, evaluation outputs, and dashboard analytics.

## Problem statement

Retail demand changes across stores, products, dates, promotions, holidays, activity, and weather. This project treats daily sales estimation as a supervised time-series regression problem for individual store-product series.

## Key features

- Exploratory data analysis and validation.
- Calendar, lag, and rolling-mean feature engineering.
- Previous-day and previous-week baselines.
- LightGBM and LSTM training and evaluation.
- Recursive 1–14-day forecasting using the saved LightGBM model, with a 7-day default.
- Streamlit dashboard with Plotly charts, accuracy diagnostics, and historical reliability analysis.
- Optional CSV upload for incorporating newly arrived raw data without retraining.

## Methodology and workflow

1. Load `data/raw/train (1).parquet`.
2. Inspect and validate dates, identifiers, missing values, duplicates, sales, stock, weather, and store-product coverage.
3. Engineer calendar features plus sales lags (`1`, `7`, `14`, and `28` days) and rolling means (`7`, `14`, and `28` days).
4. Save the engineered data as `data/processed/sales_feature_engineered.parquet`.
5. Use a chronological train/test split at `2025-02-10`.
6. Compare lag baselines, LightGBM, and LSTM models.
7. Generate future predictions by rolling the LightGBM model forward recursively.
8. Review forecasts and diagnostics in the Streamlit dashboard.

## Forecasting approach and models

### Baselines

- **Lag 1:** uses the previous day's sales.
- **Lag 7:** uses sales from the previous week.

### LightGBM

The saved LightGBM model uses categorical identifiers, operational and weather features, calendar features, and sales-history features. Recorded feature importance places `store_id`, `product_id`, `lag_1`, `day_of_month`, `avg_temperature`, and `lag_7` among the leading features.

### LSTM

The LSTM uses a 15-day input window, scaled continuous features, and embeddings for city, store, and product identifiers. The recorded run completed three epochs.

### Recursive forecasting

`models/forecast_pipeline.py` converts the one-step LightGBM model into a multi-day forecaster: each prediction is added to the running history used to calculate later lag and rolling features. Future calendar values are derived from the date; other future context remains based on the latest known values unless overrides are supplied.

## Data description

The expected source dataset contains 7,869,549 rows and 27 columns covering `2023-06-05` through `2025-07-13`, according to `models/run_metadata.json`. The raw and engineered Parquet files are not included in this checkout because Parquet files are ignored by `.gitignore`.

The data includes store, product, category, date, sales, stock, activity, discount, holiday, precipitation, temperature, humidity, and wind fields. The tracked samples are:

- `data/processed/retail_sample.csv` — 10,000-row raw-data sample.
- `data/processed/sales_feature_engineered_sample100.csv` — 100-row engineered sample.

`stock_hour6_22_cnt` is retained in the data but excluded from the LightGBM feature list.

## Dashboard and analytics

`models/dashboard.py` provides:

- Store and product selection.
- Next-day, next-7-day, or custom 1–14-day forecasts.
- Historical actuals and future recursive forecasts in a Plotly chart.
- Optional historical model predictions versus actuals.
- Overall LightGBM test R² and cumulative forecast sales.
- Selected-pair MAE, RMSE, R², normalized error, and reliability label.
- Forecast reliability ranking by historical error and sales volatility.
- Optional forecast-based replenishment indicator using user-entered inventory and lead-time assumptions.

The replenishment indicator is a proxy, not an actual inventory, reorder-point, or stockout prediction. Reliability rankings are historical diagnostics, not live risk signals. The analytics module's residual-based band is also an empirical proxy, not a confidence or guaranteed prediction interval.

## Recorded results

The following values are recorded in `models/final_model_comparison.csv`:

| Model | Split | MAE | RMSE | R² | Observations |
|---|---|---:|---:|---:|---:|
| Lag 1 baseline | Test | 0.5999 | 0.9442 | 0.5100 | 1,639,185 |
| Lag 7 baseline | Test | 0.6287 | 1.0089 | 0.4443 | 1,612,608 |
| **LightGBM (full test set)** | **Test** | **0.4705** | **0.7263** | **0.7096** | **1,643,401** |
| LightGBM (matched to LSTM rows) | Test | 0.4712 | 0.7279 | 0.7137 | 1,576,934 |
| LSTM | Validation | 0.7467 | 1.0362 | 0.3090 | 1,215,600 |
| LSTM | Test | 0.8561 | 1.2689 | 0.1301 | 1,576,934 |

On the recorded full test set, LightGBM has lower MAE and RMSE than either lag baseline and an R² of `0.7096`, compared with `0.5100` for the lag-1 baseline and `0.1301` for the LSTM test result. The separate LightGBM evaluation matched to the LSTM rows reports R² `0.7137`.

## Project structure

```text
data/
└── processed/
    ├── retail_sample.csv
    └── sales_feature_engineered_sample100.csv

models/
├── analytics.py
├── dashboard.py
├── forecast_pipeline.py
├── best_lightgbm.pkl
├── best_lstm.pth
├── lgbm_features.pkl
├── lstm_scaler.pkl
├── test_predictions.csv
├── final_model_comparison.csv
├── lightgbm_feature_importance.csv
└── run_metadata.json

notebooks/
├── 1_eda.ipynb
├── 2_Preprocessing_FeatureEngg.ipynb
├── 3_Baseline.ipynb
├── inspect_data.py
└── validate_data.py

requirements.txt
```

## Tech stack

Python, pandas, NumPy, PyArrow, scikit-learn, LightGBM, PyTorch, Streamlit, Plotly, Joblib, Matplotlib, Seaborn, and Jupyter.

## Installation and setup

```bash
pip install -r requirements.txt
```

Add the source file at:

```text
data/raw/train (1).parquet
```

Then run the preprocessing notebook to create:

```text
data/processed/sales_feature_engineered.parquet
```

The saved model artifacts are included, but the dashboard and forecasting pipeline require the engineered Parquet history at runtime.

## How to run

Inspect and validate the source data:

```bash
python notebooks/inspect_data.py
python notebooks/validate_data.py
```

Run the forecasting smoke test:

```bash
python models/forecast_pipeline.py
```

Launch the dashboard:

```bash
streamlit run models/dashboard.py
```

## Future improvements

- Document/provide the raw and engineered Parquet inputs.
- Replace carried-forward future context with known holiday, promotion, activity, and weather values where available.
- Validate recursive feature definitions against the training workflow.
- Improve LSTM training and tuning, and expose the existing empirical uncertainty band in the dashboard with clear limitations.

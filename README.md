# Supply Chain Sales Forecasting & Analytics

A machine learning based retail sales forecasting project that predicts future product-level sales for individual stores and provides analytics for forecasting performance, reliability, and replenishment planning.

The project uses historical retail sales data along with store, product, promotional, holiday, activity, and weather-related features. A LightGBM-based forecasting pipeline is used to generate recursive multi-day forecasts, supported by an interactive Streamlit dashboard.

---

## 📌 Project Overview

Accurate sales forecasting is an important component of supply chain management. Reliable demand estimates can help businesses improve inventory planning, reduce stockouts and overstocking, and make better replenishment decisions.

This project focuses on:

- Exploring and validating a large retail sales dataset
- Performing data preprocessing and feature engineering
- Creating time-series based lag and rolling features
- Training a LightGBM regression model for sales forecasting
- Generating recursive multi-day forecasts
- Evaluating historical forecasting performance
- Estimating forecast reliability using historical residuals and sales volatility
- Providing a sales-based replenishment proxy
- Building an interactive Streamlit dashboard for prediction and analysis

The primary forecasting unit is a **store-product pair**.

---

# 📊 Dataset

## Main Dataset

The main dataset used in this project is:

```text
train (1).parquet
```

Location:

```text
data/raw/train (1).parquet
```

The dataset contains approximately:

- **7.87 million rows**
- **19 columns**
- **1,057 stores**
- **576 products**
- **18 cities**
- **22,939 store-product time series**
- Date range: **2023-06-05 to 2025-07-13**
- **770 unique dates**

The target variable for forecasting is:

```text
sale_amount
```

---

## Main Dataset Features

The dataset contains historical sales and several explanatory variables.

| Feature | Description |
|---|---|
| `dt` | Date of the observation |
| `sale_amount` | Daily sales amount and forecasting target |
| `store_id` | Unique identifier of the store |
| `product_id` | Unique identifier of the product |
| `city_id` | Identifier of the city |
| `discount` | Discount-related feature |
| `activity_flag` | Indicates whether the relevant activity/promotion was active |
| `holiday_flag` | Indicates whether the date was a holiday |
| `precpt` | Precipitation/weather-related feature |
| `avg_temperature` | Average temperature |
| `avg_humidity` | Average humidity |
| `avg_wind_level` | Average wind level |
| Other categorical/metadata fields | Store/product/category related information |

The exact model-training feature set is prepared during preprocessing and feature engineering.

---

# 📁 Mini Dataset for Data Inspection

A smaller CSV dataset is also created for convenient inspection:

```text
retail_sample.csv
```

Location:

```text
data/processed/retail_sample.csv
```

This file contains the **first 10,000 rows** of the main dataset.

It is generated using:

```text
notebooks/inspect_data.py
```

The purpose of this CSV is to make the data easier to:

- Open in Excel
- Quickly inspect
- Understand the column structure
- Check sample records
- Demonstrate the dataset without loading the complete dataset

### Important

`retail_sample.csv` is **not the main training dataset**.

The original:

```text
train (1).parquet
```

remains the primary dataset used by the project.

The mini CSV is only a convenient sample for inspection and demonstration.

---

# 🔍 Data Validation

Before modelling, the dataset is checked using:

```text
notebooks/validate_data.py
```

The validation process checks important data-quality properties such as:

- Dataset dimensions
- Column structure
- Missing values
- Date validity
- Number of stores
- Number of products
- Store-city consistency
- Product-category consistency
- Duplicate store-product-date records
- Negative sales values
- Zero-sales observations
- Stockout-hour ranges
- Holiday and activity flags
- Weather-related feature ranges
- Store-product time-series coverage
- Missing dates and date gaps
- Store-date coverage

The validation showed that the checked dataset has:

- No missing values
- No duplicate store-product-date records
- No negative sales values
- Valid date values
- Valid stockout-hour ranges

Some store-product series have different lengths and date gaps. These are treated as characteristics of the available dataset and are considered during forecasting-data validation.

---

# 🧹 Data Preprocessing & Feature Engineering

The project uses time-series features to capture historical demand patterns.

## Calendar Features

Date information is transformed into useful calendar-based variables such as:

- Day-related features
- Week-related features
- Month-related features
- Other temporal information derived from the date

These features help the model learn recurring patterns in sales.

---

## Lag Features

Historical sales values are used as lag features.

The current forecasting pipeline supports the following lag requirements:

```text
lag_1
lag_7
lag_14
lag_28
```

These represent sales from:

- 1 day earlier
- 7 days earlier
- 14 days earlier
- 28 days earlier

Lag features help the model learn short-term, weekly, and longer-term demand patterns.

---

## Rolling Mean Features

Rolling historical averages are also used:

```text
rolling_mean_7
rolling_mean_14
rolling_mean_28
```

These represent historical average sales over different time windows.

Rolling features help smooth short-term fluctuations and provide the model with information about recent demand levels.

---

# 🤖 Machine Learning Model

The project uses **LightGBM** as the primary forecasting model.

The model is trained as a regression model to predict sales for a store-product combination.

The saved model artifacts include:

```text
best_lightgbm.pkl
lgbm_features.pkl
```

### `best_lightgbm.pkl`

Contains the trained LightGBM model used for generating predictions.

### `lgbm_features.pkl`

Contains the feature information required by the trained model.

The forecasting pipeline uses this saved feature list to determine the required lag and rolling-window features.

This helps keep the forecasting logic aligned with the saved model during future retraining.

---

# 🔮 Forecasting Pipeline

The main forecasting logic is implemented in:

```text
forecast_pipeline.py
```

This file provides the functionality required to generate future sales forecasts.

The pipeline performs the following major steps:

1. Load the trained LightGBM model
2. Load the saved model feature list
3. Validate the required historical data
4. Identify the required lag and rolling features
5. Prepare the latest historical observations
6. Generate features for the next date
7. Predict sales for that date
8. Add the prediction back into the historical series
9. Use the newly predicted value for subsequent predictions
10. Repeat the process for the required forecasting horizon

---

# 🔁 Recursive Forecasting

The forecasting approach is **recursive**.

Instead of directly predicting all future days at once, the model predicts one future day at a time.

For example:

```text
Historical Data
      ↓
Predict Day 1
      ↓
Add Day 1 prediction to history
      ↓
Predict Day 2
      ↓
Add Day 2 prediction to history
      ↓
Predict Day 3
      ↓
...
```

This allows the same one-step LightGBM model to be used for multiple future days.

The primary forecasting horizon used by the project is:

```text
7 days
```

The pipeline can also support other forecasting horizons.

---

# 📅 Future Feature Assumptions

Some future values are naturally known from the calendar, while others need assumptions.

Calendar features can be calculated directly for future dates.

For future explanatory variables such as:

- `discount`
- `activity_flag`
- `holiday_flag`
- Weather-related variables

the forecasting pipeline can carry forward the latest available values unless future values are explicitly provided as overrides.

Therefore, future forecasting should be interpreted as a practical forecasting scenario based on the available information and these assumptions.

Forecast reliability generally decreases as the recursive horizon increases because predictions from earlier future days become inputs to later predictions.

For this reason, the **7-day forecast is treated as the primary forecasting use case**.

---

# 🛠️ Dynamic Forecast Feature Requirements

The forecasting pipeline does not rely only on a permanently hardcoded list of lag and rolling features.

The required lag and rolling-window features are extracted from:

```text
lgbm_features.pkl
```

For example, if the saved model contains:

```text
lag_1
lag_7
lag_14
lag_28
rolling_mean_7
rolling_mean_14
rolling_mean_28
```

the forecasting pipeline automatically determines these requirements.

This makes the forecasting code more robust if the feature configuration changes during future model retraining.

---

# 📈 Analytics

Additional forecasting analytics are implemented in:

```text
analytics.py
```

This module provides analysis beyond the raw prediction.

## Forecast Error Analysis

Historical prediction errors are analyzed using held-out test predictions.

The project examines the difference between:

```text
Actual Sales
-
Predicted Sales
```

These residuals provide information about historical model behaviour.

---

## Forecast Band

An empirical residual-based forecast band is calculated to provide an indication of the historical spread of forecast errors.

This is **not a formal statistical confidence interval**.

It should instead be interpreted as a historical error-based range around the forecast.

---

## Store-Product Performance

The project evaluates forecasting performance for individual store-product combinations.

This helps identify combinations where the forecasting model has historically performed:

- Relatively well
- Relatively poorly

Performance metrics are based on historical held-out predictions.

---

## Reliability Ranking

The project also generates a reliability-oriented ranking.

The ranking considers factors such as:

- Historical forecasting error
- Sales volatility

This helps distinguish store-product combinations where forecasts have historically been more stable from those where forecasting has been more difficult.

The reliability ranking is **historical** and should not be interpreted as a live stockout-risk or real-time operational risk score.

---

# 📦 Replenishment Proxy

The project also provides a replenishment-oriented analytical view.

The available dataset does not contain complete operational inventory information such as:

- Actual inventory level
- Supplier lead time
- Reorder point
- Safety stock
- Purchase orders

Therefore, the project does **not** calculate a true inventory optimization policy.

Instead, the dashboard provides a **sales-based replenishment proxy** that can help identify products/stores with stronger expected demand.

This can be useful as a starting point for supply-chain analysis.

---

# 📊 Interactive Dashboard

The project includes an interactive Streamlit dashboard:

```text
dashboard.py
```

The dashboard provides a user-friendly interface for exploring forecasts and historical performance.

The dashboard allows users to select:

- Store
- Product
- Forecast horizon

---

# 🔮 Forecast Horizon Options

The dashboard supports:

```text
Next day
Next 7 days
Custom horizon
```

The custom forecasting horizon can be selected within the supported range of:

```text
1–14 days
```

The 7-day forecast is the primary recommended horizon.

---

# 📋 Dashboard Sections

## 1. Forecast

Displays future predicted sales for the selected store-product pair.

The forecast includes the selected future dates and predicted sales values.

---

## 2. Historical Diagnostics

Historical sales and model behaviour can be inspected for the selected store-product combination.

This helps users understand the demand pattern before interpreting the future forecast.

---

## 3. Performance

Historical forecasting performance is displayed using held-out test predictions.

This provides context about how well the model has performed for the selected store-product combination.

---

## 4. Reliability

The dashboard provides a historical reliability view based on forecasting performance and sales variability.

This helps users identify store-product combinations where forecasting is relatively more or less reliable.

---

## 5. Replenishment Proxy

A sales-based replenishment proxy is provided to support supply-chain interpretation.

It should be treated as an analytical indicator rather than a complete inventory-replenishment algorithm.

---

# 📤 Latest Data Upload

The dashboard also supports uploading a latest-data CSV without retraining the model.

The uploaded file should contain the required raw input fields, including:

```text
dt
sale_amount
store_id
product_id
city_id
discount
activity_flag
holiday_flag
precpt
avg_temperature
avg_humidity
avg_wind_level
```

The user does not need to manually provide:

- Lag features
- Rolling features
- Calendar features

These are computed automatically by the forecasting pipeline.

This makes the dashboard easier to use with updated data.

---

# 📁 Project Structure

```text
SupplyChainForecasting/
│
├── data/
│   ├── raw/
│   │   └── train (1).parquet
│   │
│   └── processed/
│       ├── retail_sample.csv
│       └── sales_feature_engineered.parquet
│
├── notebooks/
│   ├── 1_eda.ipynb
│   ├── 2_Preprocessing_FeatureEngg.ipynb
│   ├── 3_Baseline.ipynb
│   ├── inspect_data.py
│   └── validate_data.py
│
├── models/
│   ├── best_lightgbm.pkl
│   └── lgbm_features.pkl
│
├── analytics.py
├── forecast_pipeline.py
├── dashboard.py
├── requirements.txt
├── .gitignore
└── README.md
```

> Note: The exact files present in the GitHub repository may depend on which generated datasets and model artifacts are uploaded.

---

# 📂 File-by-File Description

## `README.md`

Provides the complete documentation of the project, including:

- Project objective
- Dataset information
- Feature engineering
- Machine learning approach
- Forecasting pipeline
- Analytics
- Dashboard
- Installation
- Usage
- Limitations

---

## `requirements.txt`

Contains the Python libraries required to run the project.

Major dependencies include:

- Pandas
- NumPy
- PyArrow
- Scikit-learn
- LightGBM
- PyTorch
- Streamlit
- Plotly
- Joblib
- Matplotlib
- Seaborn
- Jupyter

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

# 📓 Notebooks

## `1_eda.ipynb`

This notebook is used for exploratory data analysis.

It helps understand:

- Dataset structure
- Sales behaviour
- Store and product distributions
- Temporal patterns
- Feature distributions
- General characteristics of the retail data

---

## `2_Preprocessing_FeatureEngg.ipynb`

This notebook focuses on preparing the dataset for machine learning.

The main tasks include:

- Data preprocessing
- Date processing
- Feature creation
- Lag feature generation
- Rolling feature generation
- Preparation of model-ready data

---

## `3_Baseline.ipynb`

This notebook contains the baseline modelling workflow.

It is used to develop and evaluate the forecasting model before integrating the final model into the reusable forecasting pipeline and dashboard.

---

# 📝 `inspect_data.py`

This script is used for quick dataset inspection.

It:

1. Loads the main Parquet dataset
2. Prints dataset shape
3. Prints column names
4. Displays sample rows
5. Displays data types
6. Checks missing values
7. Creates a 10,000-row CSV sample

The generated sample is:

```text
data/processed/retail_sample.csv
```

This file is particularly useful for quickly viewing the dataset in Excel or other spreadsheet software.

---

# 🧪 `validate_data.py`

This script performs detailed data-quality validation.

It checks:

- Missing values
- Invalid dates
- Duplicate records
- Negative sales
- Store and product consistency
- Time-series coverage
- Date gaps
- Weather feature ranges
- Holiday/activity flags
- Store-date coverage
- Store-product series lengths

It is intended to verify the dataset before using it for forecasting.

---

# 🤖 `forecast_pipeline.py`

This is the main reusable forecasting module.

Its responsibilities include:

- Loading saved model artifacts
- Validating historical data
- Detecting model feature requirements
- Creating lag features
- Creating rolling features
- Generating recursive forecasts
- Handling future feature overrides
- Returning predictions in a structured format

The pipeline is designed so that forecasting can be performed without rerunning the complete model-training workflow.

---

# 📈 `analytics.py`

This module provides analytical functions used by the dashboard.

Its responsibilities include:

- Forecast error analysis
- Empirical residual-based forecast bands
- Store-product performance analysis
- Reliability ranking
- Sales volatility analysis
- Replenishment-oriented proxy calculations

The analytics are based primarily on historical model predictions and available sales data.

---

# 🖥️ `dashboard.py`

This is the Streamlit application for interacting with the forecasting system.

The dashboard connects the trained model, forecasting pipeline, historical data, and analytics into one interface.

Main capabilities include:

- Store selection
- Product selection
- Next-day forecasting
- 7-day forecasting
- Custom 1–14 day forecasting
- Historical diagnostics
- Forecast performance analysis
- Reliability analysis
- Replenishment proxy
- Latest-data CSV upload

---

# 🔄 Overall Project Workflow

The complete workflow can be summarized as:

```text
Raw Retail Dataset
        │
        ▼
Data Inspection
(inspect_data.py)
        │
        ▼
Data Validation
(validate_data.py)
        │
        ▼
Preprocessing
        │
        ▼
Feature Engineering
(Calendar + Lag + Rolling Features)
        │
        ▼
Model Training
(LightGBM)
        │
        ▼
Saved Model Artifacts
(best_lightgbm.pkl)
        │
        ▼
Forecasting Pipeline
(forecast_pipeline.py)
        │
        ▼
Recursive Future Forecast
        │
        ├───────────────┐
        ▼               ▼
   Analytics       Streamlit Dashboard
(analytics.py)       (dashboard.py)
        │               │
        └───────┬───────┘
                ▼
       Forecast & Supply
       Chain Insights
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd SupplyChainForecasting
```

---

## 2. Create a Virtual Environment

It is recommended to use a virtual environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Dashboard

From the project root directory, run:

```bash
streamlit run dashboard.py
```

The Streamlit application will open in the browser.

The dashboard requires the trained model artifacts and processed data expected by the application.

---

# 📌 Running the Data Inspection Script

From the project root:

```bash
python notebooks/inspect_data.py
```

This will:

- Load the raw dataset
- Display basic information
- Check missing values
- Create the 10,000-row sample CSV

The generated file will be:

```text
data/processed/retail_sample.csv
```

---

# 📌 Running Data Validation

From the project root:

```bash
python notebooks/validate_data.py
```

This performs the data-quality checks described earlier.

---

# 📦 Data Availability

The complete raw dataset is large and stored as a Parquet file:

```text
data/raw/train (1).parquet
```

For GitHub distribution, Parquet files are excluded using:

```gitignore
*.parquet
```

Therefore, the main raw dataset and other Parquet datasets should not be expected to be available directly after cloning the repository unless they are provided separately.

The smaller:

```text
data/processed/retail_sample.csv
```

can be included in the repository for demonstration and inspection.

---

# ⚠️ Limitations

The project has several practical limitations.

### 1. Recursive Forecasting

The forecasting pipeline predicts future dates recursively. As the forecast horizon increases, prediction errors can propagate through later predictions.

Therefore, shorter horizons are generally more reliable.

---

### 2. Future Feature Assumptions

Future discount, activity, holiday, and weather-related values may not always be known.

When explicit future values are not provided, the pipeline uses the latest available values as assumptions.

---

### 3. Replenishment Proxy

The replenishment component is not a full inventory optimization system because the dataset does not provide complete inventory, lead-time, safety-stock, or supplier information.

The replenishment output should therefore be interpreted as a sales-based proxy.

---

### 4. Reliability Interpretation

The reliability ranking is based on historical forecasting performance and sales variability.

It is not a real-time probability of stockout, demand risk, or business failure.

---

### 5. Forecast Band Interpretation

The empirical forecast band is derived from historical residuals.

It should not be interpreted as a formal statistical confidence interval or guaranteed prediction interval.

---

### 6. Dataset Coverage

Different store-product combinations have different historical lengths and some series contain date gaps.

The forecasting pipeline therefore validates the available history before attempting prediction.

---

# 💡 Key Project Features

- Large-scale retail sales forecasting
- Store-product level demand prediction
- LightGBM regression model
- Time-series feature engineering
- Calendar features
- Lag features
- Rolling mean features
- Recursive multi-day forecasting
- Dynamic model feature requirement detection
- Historical forecasting error analysis
- Empirical residual-based forecast bands
- Store-product performance analysis
- Reliability ranking
- Sales-based replenishment proxy
- Interactive Streamlit dashboard
- Latest-data CSV upload
- Data inspection and validation utilities

---

# 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data processing and analysis |
| NumPy | Numerical operations |
| PyArrow | Parquet data handling |
| Scikit-learn | Machine learning utilities and evaluation |
| LightGBM | Sales forecasting model |
| Joblib | Saving and loading model artifacts |
| Matplotlib | Visualization |
| Seaborn | Exploratory visualization |
| Plotly | Interactive visualizations |
| Streamlit | Interactive dashboard |
| Jupyter Notebook | EDA, preprocessing, and modelling workflow |

---

# 🎯 Project Objective

The main objective of this project is to build an end-to-end retail sales forecasting system that connects:

```text
Data
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Machine Learning
   ↓
Forecasting
   ↓
Performance Analytics
   ↓
Supply Chain Insights
   ↓
Interactive Dashboard
```

Rather than providing only a machine learning prediction, the project combines forecasting with analytical tools that can help users understand forecast behaviour and use the results from a supply-chain perspective.

---

# 📌 Conclusion

This project demonstrates an end-to-end machine learning workflow for retail demand forecasting.

It combines large-scale data processing, time-series feature engineering, LightGBM modelling, recursive forecasting, historical performance analysis, reliability assessment, and interactive visualization.

The final system provides a practical interface through which users can select a store-product pair, generate future sales forecasts, examine historical model performance, understand forecast reliability, and view a sales-based replenishment proxy.

The project is designed as a foundation that can be further extended with richer inventory data, explicit supplier lead times, advanced forecasting models, real-time data pipelines, and formal inventory optimization methods.

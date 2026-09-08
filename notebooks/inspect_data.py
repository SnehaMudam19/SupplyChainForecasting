from pathlib import Path
import pandas as pd


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data" / "raw" / "train (1).parquet"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "retail_sample.csv"


# Load dataset
df = pd.read_parquet(RAW_DATA)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# Create a small CSV sample for viewing in Excel
sample = df.head(10000)
sample.to_csv(PROCESSED_DATA, index=False)

print("\n10,000-row sample saved successfully.")
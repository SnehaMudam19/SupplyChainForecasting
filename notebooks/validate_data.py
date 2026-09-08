from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "raw" / "train (1).parquet"

df = pd.read_parquet(file_path)

# --------------------------------------------------
# BASIC STRUCTURE CHECK
# --------------------------------------------------

print("\nBASIC STRUCTURE CHECK\n")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# --------------------------------------------------
# DATA CHECK
# --------------------------------------------------

print("\nDATE CHECK\n")

df["dt"] = pd.to_datetime(df["dt"])

print("Start date:", df["dt"].min())
print("End date:", df["dt"].max())

print("Total unique dates:", df["dt"].dt.date.nunique())

print("Invalid dates:", df["dt"].isna().sum())

# --------------------------------------------------
# STORE CHECK
# --------------------------------------------------


print("\nSTORE CHECK\n")


print("Number of stores:", df["store_id"].nunique())
print("Number of cities:", df["city_id"].nunique())

store_city = df.groupby("store_id")["city_id"].nunique()

print("Stores belonging to multiple cities:",(store_city > 1).sum())

# --------------------------------------------------
# PRODUCT CHECK
# --------------------------------------------------


print("\nPRODUCT CHECK\n")


product_stores = (
    df.groupby("product_id")["store_id"]
      .nunique()
)

print("Products:", df["product_id"].nunique())

print("Products present in multiple stores:",
      (product_stores > 1).sum())

print("\nExample:")
print(product_stores.head(10))

# Same product_id with different category hierarchy
product_hierarchy = (
    df.groupby("product_id")[
        [
            "management_group_id",
            "first_category_id",
            "second_category_id",
            "third_category_id"
        ]
    ]
    .nunique()
)

inconsistent_products = product_hierarchy[
    (product_hierarchy > 1).any(axis=1)
]

print("Products with multiple category hierarchies:",
      len(inconsistent_products))

print("\nExample inconsistent products:")
print(inconsistent_products.head(10))

# --------------------------------------------------
# STORE-PRODUCT-DATE CHECK
# --------------------------------------------------


print("\nSTORE-PRODUCT-DATE CHECK\n")


duplicate_spd = df.duplicated(
    subset=["store_id", "product_id", "dt"]
).sum()

print(
    "Duplicate Store-Product-Date rows:",
    duplicate_spd
)

# --------------------------------------------------
# SALES CHECK
# --------------------------------------------------

print("\nSALES CHECK\n")


print(df["sale_amount"].describe())

print(
    "Negative sales:",
    (df["sale_amount"] < 0).sum()
)

print(
    "Zero sales:",
    (df["sale_amount"] == 0).sum()
)

# --------------------------------------------------
# STOCKOUT CHECK
# --------------------------------------------------

print("\nSTOCKOUT CHECK\n")


print(df["stock_hour6_22_cnt"].describe())

print("Stockout hours outside 0-16:",((df["stock_hour6_22_cnt"] < 0) |(df["stock_hour6_22_cnt"] > 16) ).sum())

# --------------------------------------------------
# FLAG CHECK
# --------------------------------------------------


print("\nFLAG CHECK\n")


for col in ["holiday_flag", "activity_flag"]:
    print(
        col,
        "unique values:",
        sorted(df[col].dropna().unique())
    )

# --------------------------------------------------
# DISCOUNT + WEATHER CHECK
# --------------------------------------------------


print("\nEXTERNAL FEATURES CHECK\n")

for col in [
    "discount",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level"
]:
    print(f"\n{col}:")
    print(df[col].describe())

# --------------------------------------------------
# STORE-PRODUCT HISTORY CHECK
# --------------------------------------------------


print("\nSTORE-PRODUCT HISTORY CHECK\n")


series_days = (
    df.groupby(["store_id", "product_id"])["dt"]
      .agg(["min", "max", "nunique"])
)

series_days["span_days"] = (series_days["max"] - series_days["min"]).dt.days + 1

print("Store-product series:",len(series_days))

print("\nHistory length:")
print(series_days["span_days"].describe())

print(
    "\nSeries with >= 365 days:",
    (series_days["span_days"] >= 365).sum()
)

print(
    "Series with >= 180 days:",
    (series_days["span_days"] >= 180).sum()
)

print(
    "Series with < 90 days:",
    (series_days["span_days"] < 90).sum()
)

# --------------------------------------------------
# MANAGEMENT GROUP CHECK
# --------------------------------------------------

print("\nMANAGEMENT GROUP CHECK\n")


store_mgmt = (
    df.groupby("store_id")["management_group_id"]
      .nunique()
)

print(
    "Stores with multiple management groups:",
    (store_mgmt > 1).sum()
)

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)

# --------------------------------------------------
# DATE CONTINUITY & STORE COVERAGE CHECK
# --------------------------------------------------

print("\nDATE CONTINUITY & STORE COVERAGE CHECK\n")


# Make sure dates are datetime
df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

print("Invalid dates:", df["dt"].isna().sum())

# ==================================================
# 1. STORE-PRODUCT DATE GAPS
# ==================================================

df = df.sort_values(["store_id", "product_id", "dt"])

previous_date = (df.groupby(["store_id", "product_id"])["dt"].shift(1))

product_date_gap = (df["dt"] - previous_date).dt.days

product_gaps = product_date_gap[product_date_gap > 1]

print("\n--- STORE-PRODUCT LEVEL ---")

print("Total Store-Product gaps:", len(product_gaps))

print("Largest Store-Product gap:", product_gaps.max())

gap_series = df.loc[product_gaps.index,["store_id", "product_id"]].drop_duplicates()

print("Store-Product series with gaps:",len(gap_series))


# ==================================================
# 2. STORE-DATE COVERAGE
# ==================================================

store_date = (
    df.groupby(["store_id", "dt"])
      .size()
      .reset_index(name="records")
)

print("\n--- STORE LEVEL ---")

print("Store-Date combinations:", len(store_date))

print("Minimum records for any Store-Date:", store_date["records"].min())

print("Maximum records for any Store-Date:", store_date["records"].max())


# ==================================================
# 3. CHECK WHETHER A STORE HAS COMPLETE DATE RANGE
# ==================================================

overall_start = df["dt"].min()
overall_end = df["dt"].max()

expected_days = (overall_end - overall_start).days + 1

print("\nOverall date range:",overall_start, "to", overall_end)

print("Expected calendar days:",expected_days)


store_history = (
    df.groupby("store_id")["dt"]
      .agg(["min", "max", "nunique"])
)

store_history["span_days"] = (
    store_history["max"] -
    store_history["min"]
).dt.days + 1

store_history["missing_days"] = (
    store_history["span_days"] -
    store_history["nunique"]
)

print("\nStores with missing dates inside their own range:",
      (store_history["missing_days"] > 0).sum())

print("Stores covering entire dataset period:",
      (
          (store_history["min"] == overall_start) &
          (store_history["max"] == overall_end)
      ).sum())


# ==================================================
# 4. DATES WITH VERY FEW STORES
# ==================================================

date_store_count = ( df.groupby("dt")["store_id"].nunique())

print("\n--- DATE LEVEL ---")

print("Maximum stores recorded on a date:",date_store_count.max())

print("Minimum stores recorded on a date:",date_store_count.min())

print("Dates with fewer than 90% of stores:",
      ( date_store_count <   0.9 * df["store_id"].nunique()).sum())


# ==================================================
# 5. STORE-DATE RECORD COUNTS
# ==================================================

print("\nStore-Date record count statistics:")
print(store_date["records"].describe())



print("\nDATE COVERAGE CHECK COMPLETE\n")

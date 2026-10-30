import pandas as pd

RAW = "data/raw/city_day.csv"
OUT = "data/processed/clean.csv"
POLLUTANTS = ["PM2.5", "PM10", "NO2", "CO"]

df = pd.read_csv(RAW, parse_dates=["Date"])
df = df[["City", "Date"] + POLLUTANTS + ["AQI", "AQI_Bucket"]]

# City-wise, date order la sort (lag features ku mukkiyam)
df = df.sort_values(["City", "Date"]).reset_index(drop=True)

# 1. Missing pollutant values: city-wise forward fill (max 7 days)
df[POLLUTANTS] = df.groupby("City")[POLLUTANTS].ffill(limit=7)

# 2. Innum missing irundha: city median, adhuvum illana overall median
for col in POLLUTANTS:
    df[col] = df[col].fillna(df.groupby("City")[col].transform("median"))
    df[col] = df[col].fillna(df[col].median())

# 3. Outliers: 99.5th percentile ku cap
for col in POLLUTANTS:
    cap = df[col].quantile(0.995)
    df[col] = df[col].clip(upper=cap)
    print(f"{col} capped at {cap:.2f}")

# 4. Label illaadha rows drop
before = len(df)
df = df.dropna(subset=["AQI_Bucket"]).reset_index(drop=True)
print(f"Dropped {before - len(df)} rows without label")

print("\nMissing values after cleaning:")
print(df.isnull().sum())

df.to_csv(OUT, index=False)
print("\nSaved:", OUT, df.shape)
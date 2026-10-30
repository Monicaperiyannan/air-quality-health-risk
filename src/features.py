import pandas as pd

IN = "data/processed/labeled.csv"
OUT = "data/processed/features.csv"
POLLUTANTS = ["PM2.5", "PM10", "NO2", "CO"]

df = pd.read_csv(IN, parse_dates=["Date"])
df = df.sort_values(["City", "Date"]).reset_index(drop=True)

g = df.groupby("City")

for col in POLLUTANTS:
    # Lag features: 1 day and 3 days munnadi value
    df[f"{col}_lag1"] = g[col].shift(1)
    df[f"{col}_lag3"] = g[col].shift(3)

    # Rolling averages: last 3 and 7 days (indha naal ah serkaama, shift(1) pannitu)
    df[f"{col}_roll3"] = g[col].transform(lambda s: s.shift(1).rolling(3).mean())
    df[f"{col}_roll7"] = g[col].transform(lambda s: s.shift(1).rolling(7).mean())

# Lag/rolling NaN irukkura first rows drop
before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"Dropped {before - len(df)} rows (initial NaNs from lag/rolling)")

# Date features
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek

# AQI and AQI_Bucket features-a use pannakoodadhu (leakage), so ippove save la irundhu theriya vekkaren
print("Columns:", df.columns.tolist())
print(df["Risk"].value_counts())

df.to_csv(OUT, index=False)
print("\nSaved:", OUT, df.shape)
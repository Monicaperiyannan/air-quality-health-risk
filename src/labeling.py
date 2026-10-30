import pandas as pd

IN = "data/processed/clean.csv"
OUT = "data/processed/labeled.csv"

df = pd.read_csv(IN, parse_dates=["Date"])

mapping = {
    "Good": "Good",
    "Satisfactory": "Good",
    "Moderate": "Moderate",
    "Poor": "Poor",
    "Very Poor": "Severe",
    "Severe": "Severe",
}
df["Risk"] = df["AQI_Bucket"].map(mapping)

# Model ku numeric label (0 to 3)
risk_order = {"Good": 0, "Moderate": 1, "Poor": 2, "Severe": 3}
df["Risk_Label"] = df["Risk"].map(risk_order)

print(df["Risk"].value_counts())
print("\nUnmapped rows:", df["Risk"].isnull().sum())

df.to_csv(OUT, index=False)
print("\nSaved:", OUT, df.shape)
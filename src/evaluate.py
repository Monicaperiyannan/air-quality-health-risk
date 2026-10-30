import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/xgb_model.pkl")
meta = joblib.load("models/meta.pkl")
features = meta["features"]

imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

print("Top 10 features:")
print(imp.head(10).round(4))

# Pollutant-wise total importance (lag, rolling ellam serthu)
def base_name(f):
    for p in ["PM2.5", "PM10", "NO2", "CO"]:
        if f.startswith(p):
            return p
    return "Other (City/Month/Day)"

by_pollutant = imp.groupby(imp.index.map(base_name)).sum().sort_values(ascending=False)
print("\nPollutant-wise importance:")
print(by_pollutant.round(4))

# Plot 1: top 15 features
plt.figure(figsize=(8, 6))
imp.head(15).sort_values().plot(kind="barh")
plt.title("Top 15 Feature Importances (XGBoost)")
plt.tight_layout()
plt.savefig("reports/figures/feature_importance.png", dpi=150)
plt.close()

# Plot 2: pollutant-wise
plt.figure(figsize=(6, 4))
by_pollutant.sort_values().plot(kind="barh")
plt.title("Pollutant-wise Importance")
plt.tight_layout()
plt.savefig("reports/figures/pollutant_importance.png", dpi=150)
plt.close()

print("\nSaved feature_importance.png and pollutant_importance.png")
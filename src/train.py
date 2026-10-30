import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

df = pd.read_csv("data/processed/features.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# City ah number ah maathrom (mapping save panrom, app ku venum)
cities = sorted(df["City"].unique())
city_map = {c: i for i, c in enumerate(cities)}
df["City_code"] = df["City"].map(city_map)

DROP = ["City", "Date", "AQI", "AQI_Bucket", "Risk", "Risk_Label"]
features = [c for c in df.columns if c not in DROP]
target = "Risk_Label"
class_names = ["Good", "Moderate", "Poor", "Severe"]

# Time-based split: first 80% dates train, last 20% test
cutoff = df["Date"].quantile(0.8)
train = df[df["Date"] <= cutoff]
test = df[df["Date"] > cutoff]
print("Cutoff date:", cutoff.date(), "| Train:", len(train), "| Test:", len(test))

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# Baseline: Logistic Regression
base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
base.fit(X_train, y_train)
base_pred = base.predict(X_test)

# XGBoost with class weights
weights = compute_sample_weight("balanced", y_train)
model = XGBClassifier(
    objective="multi:softprob",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
)
model.fit(X_train, y_train, sample_weight=weights)
pred = model.predict(X_test)

print("\n=== Baseline (Logistic Regression) ===")
print("Macro F1   :", round(f1_score(y_test, base_pred, average="macro"), 4))
print("Weighted F1:", round(f1_score(y_test, base_pred, average="weighted"), 4))

print("\n=== XGBoost ===")
print("Macro F1   :", round(f1_score(y_test, pred, average="macro"), 4))
print("Weighted F1:", round(f1_score(y_test, pred, average="weighted"), 4))
print("\n", classification_report(y_test, pred, target_names=class_names))

# Confusion matrix save
cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/figures/confusion_matrix.png", dpi=150)
plt.close()

# Model + feature list save
joblib.dump(model, "models/xgb_model.pkl")
joblib.dump({"features": features, "city_map": city_map, "class_names": class_names}, "models/meta.pkl")
print("\nSaved model and confusion matrix")
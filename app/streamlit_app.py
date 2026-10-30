import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ICONS = {"Good": "🟢", "Moderate": "🟡", "Poor": "🟠", "Severe": "🔴"}

st.set_page_config(page_title="Air Quality Health Risk", page_icon="🌫️")


@st.cache_resource
def load_model():
    model = joblib.load(BASE / "models" / "xgb_model.pkl")
    meta = joblib.load(BASE / "models" / "meta.pkl")
    return model, meta


@st.cache_data
def load_data():
    return pd.read_csv(BASE / "data" / "processed" / "features.csv", parse_dates=["Date"])


model, meta = load_model()
features = meta["features"]
city_map = meta["city_map"]
class_names = meta["class_names"]


def show_result(row):
    X = pd.DataFrame([row])[features]
    proba = model.predict_proba(X)[0]
    pred = class_names[int(proba.argmax())]
    st.subheader(f"{ICONS[pred]} Predicted Risk: {pred}")
    st.bar_chart(pd.Series(proba, index=class_names, name="Probability"))
    return pred


st.title("🌫️ Air Quality Health Risk Classifier")
st.caption("Predicts Good / Moderate / Poor / Severe health risk from PM2.5, PM10, NO2 and CO")

tab1, tab2 = st.tabs(["Manual Predict", "Historical Check"])

# ---------- Tab 1: manual input ----------
with tab1:
    city = st.selectbox("City", sorted(city_map.keys()), key="manual_city")
    month = st.slider("Month", 1, 12, 1)
    dow = st.selectbox("Day of week (0=Mon ... 6=Sun)", list(range(7)))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Today**")
        pm25 = st.number_input("PM2.5", 0.0, 400.0, 60.0)
        pm10 = st.number_input("PM10", 0.0, 500.0, 110.0)
        no2 = st.number_input("NO2", 0.0, 150.0, 25.0)
        co = st.number_input("CO", 0.0, 50.0, 1.0)
    with c2:
        st.markdown("**Yesterday**")
        pm25_y = st.number_input("PM2.5 ", 0.0, 400.0, 60.0)
        pm10_y = st.number_input("PM10 ", 0.0, 500.0, 110.0)
        no2_y = st.number_input("NO2 ", 0.0, 150.0, 25.0)
        co_y = st.number_input("CO ", 0.0, 50.0, 1.0)

    if st.button("Predict risk"):
        today = {"PM2.5": pm25, "PM10": pm10, "NO2": no2, "CO": co}
        yest = {"PM2.5": pm25_y, "PM10": pm10_y, "NO2": no2_y, "CO": co_y}
        row = {"Month": month, "DayOfWeek": dow, "City_code": city_map[city]}
        for p in today:
            row[p] = today[p]
            row[f"{p}_lag1"] = yest[p]
            # Simplification: older history = yesterday's value
            row[f"{p}_lag3"] = yest[p]
            row[f"{p}_roll3"] = yest[p]
            row[f"{p}_roll7"] = yest[p]
        show_result(row)
        st.caption("Note: 3-day and 7-day history is approximated using yesterday's values.")

# ---------- Tab 2: historical check ----------
with tab2:
    data = load_data()
    city_h = st.selectbox("City", sorted(data["City"].unique()), key="hist_city")
    cdf = data[data["City"] == city_h].sort_values("Date")
    date = st.selectbox("Date", cdf["Date"].dt.date.tolist()[::-1])
    r = cdf[cdf["Date"].dt.date == date].iloc[0].to_dict()
    r["City_code"] = city_map[city_h]

    pred = show_result(r)
    actual = r["Risk"]
    if pred == actual:
        st.success(f"Actual risk: {ICONS[actual]} {actual} (match)")
    else:
        st.warning(f"Actual risk: {ICONS[actual]} {actual} (mismatch)")

    st.markdown("**PM2.5 trend (last 30 days up to selected date)**")
    trend = cdf[cdf["Date"].dt.date <= date].tail(30).set_index("Date")["PM2.5"]
    st.line_chart(trend)
# Air Quality Health Risk Classifier

Multi-class classifier that predicts health risk level (**Good / Moderate / Poor / Severe**) from air quality sensor data (PM2.5, PM10, CO, NO2), deployed as a live Streamlit dashboard.

**Live Demo:** https://monica-air-quality-risk.streamlit.app

**Tech Stack:** Python, Pandas, Scikit-learn, XGBoost, Matplotlib, Streamlit

## Highlights
- Engineered time-series features (lag and rolling averages) city-wise to capture pollution trends
- Handled sensor outliers (99.5th percentile capping) and class imbalance (class weights)
- Time-based train/test split to avoid data leakage
- XGBoost: Macro F1 ~0.86, Weighted F1 ~0.88 (baseline Logistic Regression: Macro F1 ~0.81)
- Feature importance: PM2.5 (~44%), CO (~23%), PM10 (~18%)
- Live Streamlit dashboard with manual prediction and historical check

## Dataset
Air Quality Data in India (2015-2020), Kaggle: `rohanrao/air-quality-data-in-india`, file `city_day.csv` (26 cities).

The 6 AQI buckets were mapped to 4 risk levels:

| AQI bucket | Risk level |
|---|---|
| Good, Satisfactory | Good |
| Moderate | Moderate |
| Poor | Poor |
| Very Poor, Severe | Severe |

## Pipeline
1. **Cleaning** (`src/data_cleaning.py`): city-wise forward fill, median fill, outlier capping, drop unlabeled rows
2. **Labeling** (`src/labeling.py`): map 6 AQI buckets to 4 risk classes
3. **Features** (`src/features.py`): lag (1, 3 days) and rolling mean (3, 7 days) per pollutant, month, day of week
4. **Training** (`src/train.py`): time-based split, Logistic Regression baseline vs XGBoost with balanced class weights
5. **Evaluation** (`src/evaluate.py`): feature importance and pollutant-wise importance
6. **App** (`app/streamlit_app.py`): Manual Predict and Historical Check tabs

AQI and AQI_Bucket columns are not used as features, to avoid data leakage.

## Results

| Model | Macro F1 | Weighted F1 |
|---|---|---|
| Logistic Regression (baseline) | 0.807 | 0.826 |
| **XGBoost** | **0.856** | **0.883** |

Per-class F1 (XGBoost): Good 0.93, Moderate 0.84, Poor 0.78, Severe 0.88.

Poor is the weakest class, as it is a boundary class between Moderate and Severe with fewer samples.

## Project Structure
```
air-quality-health-risk/
├── app/streamlit_app.py     # dashboard
├── data/processed/          # cleaned, labeled, feature data
├── models/                  # trained model and metadata
├── notebooks/               # EDA
├── reports/figures/         # confusion matrix, feature importance
├── src/                     # pipeline scripts
└── requirements.txt
```

## Run Locally
```bash
git clone https://github.com/Monicaperiyannan/air-quality-health-risk.git
cd air-quality-health-risk
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Limitations
- In Manual Predict, 3-day and 7-day history is approximated using yesterday's values.
- Dataset covers 2015 to mid-2020 for 26 Indian cities.
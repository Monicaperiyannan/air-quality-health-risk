# Air Quality Health Risk Classifier

Multi-class classifier that predicts health risk level (Good / Moderate / Poor / Severe) from air quality sensor data (PM2.5, PM10, CO, NO2).

**Tech Stack:** Python, Pandas, Scikit-learn, XGBoost, Matplotlib, Streamlit

## Highlights
- Engineered time-series features (lag and rolling averages) city-wise to capture pollution trends
- Handled sensor outliers (99.5th percentile capping) and class imbalance (class weights)
- Time-based train/test split to avoid data leakage
- XGBoost: Macro F1 ~0.86, Weighted F1 ~0.88 (baseline Logistic Regression: Macro F1 ~0.81)
- Feature importance: PM2.5 (~44%), CO (~23%), PM10 (~18%)
- Live Streamlit dashboard: manual prediction + historical check

## Dataset
Air Quality Data in India (2015-2020), Kaggle (rohanrao/air-quality-data-in-india), file `city_day.csv`.

## Project Structure
- `src/` : cleaning, labeling, features, training, evaluation scripts
- `app/streamlit_app.py` : dashboard
- `models/` : trained model
- `reports/figures/` : confusion matrix, feature importance

## Run Locally
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Live Demo
(link will be added after deployment)
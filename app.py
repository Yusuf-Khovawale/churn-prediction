import streamlit as st
import pandas as pd
import numpy as np
import json, os, joblib

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Predictor — Full Feature App")

# --- Paths (app.py is in project root) ---
MODEL_XGB_PATH = os.path.join("models", "xgb_churn.joblib")
MODEL_LOGIT_PATH = os.path.join("models", "logit_churn.joblib")
SCALER_PATH = os.path.join("models", "scaler.joblib")
FEATURES_PATH = os.path.join("models", "feature_columns.json")
NUMCOLS_PATH = os.path.join("models", "numeric_columns.json")

# --- Load artifacts ---
model = None
if os.path.exists(MODEL_XGB_PATH):
    model = joblib.load(MODEL_XGB_PATH)
    st.success("Loaded XGBoost model.")
elif os.path.exists(MODEL_LOGIT_PATH):
    model = joblib.load(MODEL_LOGIT_PATH)
    st.success("Loaded Logistic Regression model.")
else:
    st.error("No model found. Train & save a model in the notebook first (models/*.joblib).")
    st.stop()

scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

try:
    with open(FEATURES_PATH) as f:
        TRAIN_COLS = json.load(f)
    with open(NUMCOLS_PATH) as f:
        NUM_COLS = json.load(f)
except Exception as e:
    st.error("Missing feature_columns.json / numeric_columns.json. Re-run the notebook cell that saves them.")
    st.stop()

# --- Telco raw input schema (from dataset) ---
st.markdown("### Customer details")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("SeniorCitizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    phone = st.selectbox("PhoneService", ["No", "Yes"])
    multiple = st.selectbox("MultipleLines", ["No phone service", "No", "Yes"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])

with col2:
    online_sec = st.selectbox("OnlineSecurity", ["No internet service", "No", "Yes"])
    online_bak = st.selectbox("OnlineBackup", ["No internet service", "No", "Yes"])
    device_prot = st.selectbox("DeviceProtection", ["No internet service", "No", "Yes"])
    tech_support = st.selectbox("TechSupport", ["No internet service", "No", "Yes"])
    stream_tv = st.selectbox("StreamingTV", ["No internet service", "No", "Yes"])
    stream_mv = st.selectbox("StreamingMovies", ["No internet service", "No", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("PaperlessBilling", ["No", "Yes"])

pay_method = st.selectbox(
    "PaymentMethod",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=300.0, value=70.0, step=0.1)
total = st.number_input("TotalCharges", min_value=0.0, max_value=10000.0, value=float(monthly*tenure), step=1.0)

# Compute engineered feature like in notebook
avg_spend = total / (tenure if tenure != 0 else 1)

# --- Build a single-row raw DataFrame (pre-encoding) ---
raw = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": online_sec,
    "OnlineBackup": online_bak,
    "DeviceProtection": device_prot,
    "TechSupport": tech_support,
    "StreamingTV": stream_tv,
    "StreamingMovies": stream_mv,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": pay_method,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "AvgMonthlySpend": avg_spend
}])

st.divider()
st.caption("We will now apply the **same preprocessing** as training (one-hot with drop_first=True), then align columns.")

# --- Apply the same one-hot encoding as training ---
X_input = pd.get_dummies(raw, drop_first=True)

# Add any missing columns (that existed in training) with 0; drop any extras; order columns
missing_cols = [c for c in TRAIN_COLS if c not in X_input.columns]
for c in missing_cols:
    X_input[c] = 0

extra_cols = [c for c in X_input.columns if c not in TRAIN_COLS]
if extra_cols:
    X_input = X_input.drop(columns=extra_cols)

X_input = X_input[TRAIN_COLS]  # exact order

# --- Scale numeric columns if your final model was trained with scaling (e.g., Logistic) ---
# For XGBoost this is not necessary, but harmless if applied consistently.
if scaler is not None:
    # Only transform columns that were numeric in training
    cols_to_scale = [c for c in NUM_COLS if c in X_input.columns]
    X_input[cols_to_scale] = scaler.transform(X_input[cols_to_scale])

if st.button("Predict churn probability"):
    try:
        proba = float(model.predict_proba(X_input)[:, 1][0])
        st.metric("Estimated churn probability", f"{proba:.2f}")
        st.success("Prediction complete ✓")
        with st.expander("See encoded feature vector (debug)", expanded=False):
            st.write(X_input)
    except Exception as e:
        st.error(f"Prediction failed. Details: {e}")
        st.stop()

st.divider()
st.markdown(
    "✅ **This app mirrors training preprocessing**: one-hot encoding with `drop_first=True`, feature alignment, "
    "and numeric scaling via the saved `scaler`. Keeping feature order identical to training avoids mismatches."
)
